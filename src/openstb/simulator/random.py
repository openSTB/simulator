# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Support for random number generation."""

from dask.utils import SerializableLock
import numpy as np

from openstb.i18n.support import translations

_ = translations.load("openstb.simulator").gettext


class _BlockRNG:
    """A block of random numbers that can be generated on demand.

    This is part of the implementation of the [openstb.simulator.util.ChunkedRNG][]
    class and should not be used directly.

    """

    def __init__(self, rng: np.random.Generator, method: str, samples_per_item: int):
        """
        Parameters
        ----------
        rng
            The underlying random number generator in its initial state.
        method
            The name of the sampling method to use. This can be the name of any method
            supported by the [numpy.random.Generator][] class.
        samples_per_item
            How many samples each item needs.

        """
        self.samples_per_item = samples_per_item

        # Store the RNG and its initial state.
        self._rng = rng
        self._rng_init = dict(self._rng.bit_generator.state)

        # Find the sampling method to use.
        self._method = getattr(self._rng, method)

        # Initialise state variables.
        self._rng_lock = SerializableLock()
        self._rng_position = -1

    def sample(self, start_index: int, count: int, *args, **kwargs) -> np.ndarray:
        """Sample the random number generator.

        Note that no error checking is performed on the inputs.

        Parameters
        ----------
        start_index
            The index of the first item in the block to sample. Cannot be negative.
        count
            The number of items to sample. Must be at least one.
        *args
            Passed to the selected sampling method of the RNG.
        **kwargs
            Passed to the selected sampling method of the RNG.

        Returns
        -------
        samples : np.ndarray
            The sampled values.

        """
        with self._rng_lock:
            # The last generated position must be before start_index. If not, reset the
            # BitGenerator state.
            if self._rng_position >= start_index:
                self._rng.bit_generator.state = dict(self._rng_init)
                self._rng_position = -1

            # Advance the RNG until the last generated position is just before the start
            # of this chunk. Note that we cannot use the advance() method of the
            # underlying BitGenerator as there is not a guaranteed mapping between the
            # number of samples created and the number of bit generations.
            discard = (start_index - 1) - self._rng_position
            if discard:
                self._method(*args, **kwargs, size=discard * self.samples_per_item)
                self._rng_position += discard

            # Get the required number of samples.
            samples = self._method(*args, **kwargs, size=count * self.samples_per_item)
            self._rng_position += count

        # Reshape. We do this rather than passing a 2D size to the sampling method so we
        # can guarantee how the samples are ordered. If the generator reshapes in a
        # different way, the values for a given index may depend on the size of the
        # retrieved chunk.
        return samples.reshape(count, self.samples_per_item)


class ChunkedRNG:
    """Chunked access to a set of random samples.

    For some plugins, particularly targets, a large number of random samples may be
    required. To avoid loading them all into memory, they need to be accessed in chunks.
    In some cases, it may be required to access chunks repeatedly or in a non-sequential
    order.

    This class wraps a standard [numpy.random.Generator][] instance to provide on-demand
    access to chunks of its samples. When initialising the class, the desired
    distribution is selected. The number of samples assigned to each item is also fixed.
    For example, if used to randomise the position of targets then each item may need
    three samples, one for each coordinate.

    The samples are not cached internally; repeated requests for a chunk will compute
    the samples each time. Internally, the sample space is sub-divided into blocks of a
    fixed size. A separate sub-generator (spawned from the main generator) is used for
    each block. The states of these sub-generators are stored, and they are reset or
    advanced as necessary to generate the samples for the requested chunk. This supports
    random access, but will be most efficient if the chunks are accessed sequentially.
    Resetting from the end and repeating the sequential access (i.e., get chunks 0
    through N, then start back at 0 again) has minimal overhead for the reset. These are
    expected to be the most common patterns of use.

    """

    def __init__(self, seed: int, method: str, samples_per_item: int):
        """
        Parameters
        ----------
        seed
            The seed for the random number generator.
        method
            The name of the sampling method to use. This determines the distribution the
            samples are drawn from. It can be the name of any method supported by the
            [numpy.random.Generator][] class.
        samples_per_item
            How many random samples are assigned to each item of the output.

        """
        self.seed = seed
        self.method = method
        self.samples_per_item = samples_per_item
        self._block_size = 5_000_000

        # Create a seed sequence and bit generator for the base RNG.
        ss = np.random.SeedSequence(seed)
        bg = np.random.PCG64DXSM(ss)
        self._base_rng = np.random.default_rng(bg)
        self._base_lock = SerializableLock()

        # Ensure the sampling method is valid.
        getattr(self._base_rng, self.method)

        # Initialise the list of block generators.
        self._block_rng: list[_BlockRNG] = []

    def sample(self, start_index: int, count: int, *args, **kwargs) -> np.ndarray:
        """Sample the random number generator.

        Parameters
        ----------
        start_index
            The index of the first item to sample. Cannot be negative.
        count
            The number of items to sample. Must be at least one.

        Returns
        -------
        samples : np.ndarray
            The samples drawn from the random number generator. This will have a shape
            (count, samples_per_item).

        """
        # Check we have valid inputs.
        if start_index < 0:
            raise IndexError(_("start index of RNG chunk cannot be negative"))
        if count < 1:
            raise IndexError(_("count of RNG chunk must be at least one"))

        # Determine which blocks we are operating in.
        end_index = start_index + count - 1
        start_block = start_index // self._block_size
        end_block = end_index // self._block_size
        stop_idx = end_block + 1

        # Ensure we have spawned RNGs for these blocks.
        with self._base_lock:
            missing = stop_idx - len(self._block_rng)
            if missing:
                self._block_rng.extend(
                    _BlockRNG(subrng, self.method, self.samples_per_item)
                    for subrng in self._base_rng.spawn(missing)
                )

        # Get the samples from each of these.
        samples = []
        block_start = start_index % self._block_size
        remaining = count
        for block_idx in range(start_block, stop_idx):
            available = self._block_size - block_start
            block_count = remaining if remaining < available else available
            samples.append(
                self._block_rng[block_idx].sample(
                    block_start, block_count, *args, **kwargs
                )
            )

            # All blocks apart from the first will sample from the start of the block.
            block_start = 0
            remaining -= block_count

        return np.concat(samples)

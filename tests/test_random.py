# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from dask.distributed.utils_test import gen_cluster  # type:ignore[import-not-found]
import numpy as np
import pytest
from scipy.stats import kstest  # type:ignore[import-not-found]

from openstb.simulator import random


@pytest.mark.parametrize("method", ["uniform", "normal"])
def test_random_chunked_rng_basic(method):
    """random.ChunkedRNG: basic operation"""
    rng = random.ChunkedRNG(12345, method, 3)
    samples = rng.sample(0, 5000)
    assert samples.shape == (5000, 3)

    # Use a Kolmogorov-Smirnov test with a confidence level of 95%. Note that for the
    # kstest() function the uniform distribution is parametrised as (min, width).
    cdf = "norm" if method == "normal" else method
    stat, pvalue = kstest(samples.flat, cdf, (0, 1))
    assert pvalue > 0.05, "samples do not match the requested distribution"

    # Request the next chunk and repeat the check.
    samples2 = rng.sample(5000, 5000)
    assert samples2.shape == (5000, 3)
    stat, pvalue = kstest(samples2.flat, cdf, (0, 1))
    assert pvalue > 0.05, "samples do not match the requested distribution"

    # Ensure the samples were different.
    assert not np.any(np.isclose(samples, samples2))

    # Check accessing the same chunk again gives the same samples.
    repeat = rng.sample(5000, 5000)
    assert np.allclose(repeat, samples2)

    # Request a smaller set of samples within the first chunk and check.
    small = rng.sample(1107, 100)
    assert small.shape == (100, 3)
    assert np.allclose(small, samples[1107:1207])

    # Check a chunk overlapping the end matches in the common area.
    overlap = rng.sample(4000, 5000)
    assert overlap.shape == (5000, 3)
    assert np.allclose(overlap[:1000], samples[4000:])

    # Create another RNG with the same seed and check the results match.
    rng2 = random.ChunkedRNG(12345, method, 3)
    repeat = rng2.sample(5000, 5000)
    assert np.allclose(repeat, samples2)

    # And make sure a different seed gives different samples from the same distribution.
    rng3 = random.ChunkedRNG(54321, method, 3)
    other = rng3.sample(5000, 5000)
    assert not np.any(np.isclose(other, samples2))
    assert other.shape == (5000, 3)
    stat, pvalue = kstest(other.flat, cdf, (0, 1))
    assert pvalue > 0.05, "samples do not match the requested distribution"


def test_random_chunked_rng_boundary():
    """random.ChunkedRNG: works across block boundaries"""
    # Create an instance with smaller blocks for testing.
    rng = random.ChunkedRNG(67167109561, "uniform", 2)
    rng._block_size = 500

    # Request some samples that should go across a boundary.
    assert len(rng._block_rng) == 0
    samples = rng.sample(250, 1221)
    assert len(rng._block_rng) == 3
    assert samples.shape == (1221, 2)
    stat, pvalue = kstest(samples.flat, "uniform", (0, 1))
    assert pvalue > 0.05, "samples are not uniformly distributed"

    # Check this is repeatable.
    samples2 = rng.sample(250, 1221)
    assert np.allclose(samples2, samples)
    rng2 = random.ChunkedRNG(67167109561, "uniform", 2)
    rng2._block_size = 500
    repeat = rng2.sample(250, 1221)
    assert np.allclose(repeat, samples)


def test_random_chunked_rng_earlier_block():
    """random.ChunkedRNG: can generate from earlier blocks"""
    # Create an instance with smaller blocks for testing.
    rng = random.ChunkedRNG(641980176141, "uniform", 2)
    rng._block_size = 500

    # Request samples from three blocks.
    assert len(rng._block_rng) == 0
    rng.sample(250, 1221)
    assert len(rng._block_rng) == 3

    # Now ensure we can still sample from blocks before the last. We do this several
    # times; an earlier bug would try to call rng.spawn() with negative numbers which
    # seems to return empty lists initially and raise exceptions on subsequent calls.
    rng.sample(0, 250)
    rng.sample(500, 250)
    rng.sample(0, 250)
    rng.sample(500, 250)
    rng.sample(0, 250)
    rng.sample(500, 250)
    rng.sample(0, 250)
    rng.sample(500, 250)
    rng.sample(0, 250)
    rng.sample(500, 250)


@gen_cluster(client=True)
async def test_random_chunked_rng_dask(c, s, a, b):
    """random.ChunkedRNG: can be used in parallel operations with Dask"""
    # Create an instance with smaller blocks for testing.
    rng = random.ChunkedRNG(5679881658710, "normal", 3)
    rng._block_size = 500

    # Get some samples locally as a reference.
    samples = rng.sample(0, 2700)

    # And submit a job to our test cluster.
    futures = c.map(lambda start: rng.sample(start, 100), np.arange(0, 2700, 100))
    dask_samples = np.concat([await fut.result() for fut in futures])
    assert np.allclose(dask_samples, samples)


def test_random_chunked_rng_bounds():
    """random.ChunkedRNG: bounds checking"""
    rng = random.ChunkedRNG(17181920, "uniform", 2)

    with pytest.raises(IndexError, match="start.+negative"):
        rng.sample(-1, 200)
    with pytest.raises(IndexError, match="count.+at least one"):
        rng.sample(0, -1)

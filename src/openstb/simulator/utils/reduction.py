# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Reduction tree utilities.

Reduction trees are a common pattern in parallel processing to combine results from
multiple tasks. At each level of the tree, a set of results is aggregated to a single
result. This is then passed to the next level of the tree. For addition, instead of
computing the overall result as

    result = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8

it might be computed as

    result = ((r1 + r2) + (r3 + r4)) + ((r5 + r6) + (r7 + r8))

This reduces the number of tasks that the scheduler has to track, reducing both
scheduler overhead and the amount of memory required to store intermediate results.

In this module, the size of the tree is parametrised by the number of levels (3 in the
above example) and the number of results aggregared at each level (2 in the above
example). This leads to a *reduction factor* which is calculated as the number of
results per level raised to the power of the number of levels. In the above example this
is 2^3 = 8, meaning that 8 inputs to the tree are reduced to a single output.

"""

from typing import Any, Callable

from dask.tokenize import tokenize
import distributed


class DaskReductionTree:
    """Dask-based reduction tree to aggregate multiple futures to one.

    This is given a reduction function which is scheduled to run on the cluster. Each
    output from the tree is passed to the given output function. Note that the output
    function is called directly, not scheduled to run on the cluster.

    Note that no guarantee is made about the order the input futures are reduced in.

    """

    def __init__(
        self,
        client: distributed.Client,
        reduce_func: Callable,
        reduce_kwargs: dict[str, Any] | None,
        output_func: Callable[[distributed.Future], None],
        levels: int = 3,
        futures: int = 4,
    ):
        """
        Parameters
        ----------
        client
            The Dask client to submit reduction tasks to.
        reduce_func
            The function to call to reduce some data. This will be scheduled through the
            Dask client. It will be passed a list of the results from the input futures
            and must returned the reduced result.
        reduce_kwargs
            Keyword arguments to pass to `reduce_func`.
        output_func
            The function to call when a reduction is complete. This will be given the
            future corresponding to the final reduction and any keyword arguments.
        levels
            The number of levels of reduction to perform before calling `output_func`.
        futures
            The number of futures to reduce at each level.

        """
        self.client = client
        self.reduce_func = reduce_func
        self.reduce_kwargs = reduce_kwargs or {}
        self.output_func = output_func
        self.levels = levels
        self.futures = futures

        # Prepare storage for futures that have not been fully reduced.
        self._reducing: list[list[distributed.Future]] = []
        for i in range(self.levels):
            self._reducing.append([])

    def add_futures(self, *futures: distributed.Future) -> None:
        """Add futures to the reduction tree.

        Parameters
        ----------
        *futures
            Futures to reduce.

        """
        # Insert at the first level.
        self._reducing[0].extend(futures)

        # Process each level which has enough futures to reduce.
        for level in range(self.levels):
            while len(self._reducing[level]) >= self.futures:
                # Apply the reduction function to the first N futures.
                to_reduce = self._reducing[level][: self.futures]
                reduced = self.client.submit(
                    self.reduce_func,
                    to_reduce,
                    key=f"reduction-{level}-{tokenize(to_reduce)}",
                    **self.reduce_kwargs,
                )
                del self._reducing[level][: self.futures]

                # Output or add to the next level.
                if level == self.levels - 1:
                    self.output_func(reduced)
                else:
                    self._reducing[level + 1].append(reduced)

    def flush(self) -> None:
        """Flush the reduction tree.

        In most situations, the number of futures to reduce won't be an exact multiple
        of the reduction factor. Calling this method will reduce any futures remaining
        in the tree. There are three cases:

        * If the tree is empty, do nothing.
        * If there is one future in the entire tree, output that.
        * If there are multiple futures in the tree, call the reduction function on all
          of them and output the result.

        After this method has been called, the tree is guaranteed to be empty.

        """
        # Collect all remaining futures at any level.
        remaining = []
        for level in range(self.levels):
            remaining.extend(self._reducing[level])
            self._reducing[level].clear()

        if not remaining:
            return

        if len(remaining) == 1:
            self.output_func(remaining[0])
            return

        # Schedule the reduction and output the future.
        reduced = self.client.submit(
            self.reduce_func,
            remaining,
            key=f"reduction-finish-{tokenize(remaining)}",
            **self.reduce_kwargs,
        )
        self.output_func(reduced)

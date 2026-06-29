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

from collections.abc import Iterable
from typing import Any, Callable, Concatenate

from dask.tokenize import tokenize
import distributed

from openstb.i18n.support import translations

_ = translations.load("openstb.simulator").gettext


class DaskReductionTree[T]:
    """Dask-based reduction tree to aggregate multiple futures to one.

    This is given a reduction function which is scheduled to run on the cluster. Each
    output from the tree is passed to the given output function. Note that the output
    function is called directly, not scheduled to run on the cluster.

    Note that no guarantee is made about the order the input futures are reduced in.

    """

    def __init__(
        self,
        client: distributed.Client,
        output_func: Callable[[distributed.Future[T], Any], None],
        reduce_func: Callable[Concatenate[list[T], ...], T],
        reduce_args: Iterable[Any] | None = None,
        reduce_kwargs: dict[str, Any] | None = None,
        levels: int = 3,
        futures: int = 4,
    ):
        """
        Parameters
        ----------
        client
            The Dask client to submit reduction tasks to.
        output_func
            The function to call when a reduction is complete. This will be given the
            future corresponding to the output of the final reduction and the current
            value of the tag property.
        reduce_func
            The function to call to reduce some data. This will be scheduled through the
            Dask client. It will be passed a list of the results from the input futures
            and must returned the reduced result.
        reduce_args
            Positional arguments to pass to `reduce_func`.
        reduce_kwargs
            Keyword arguments to pass to `reduce_func`.
        levels
            The number of levels of reduction to perform before calling `output_func`.
        futures
            The number of futures to reduce at each level.

        """
        self.client = client
        self.output_func = output_func
        self.reduce_func = reduce_func
        self.reduce_args = reduce_args or []
        self.reduce_kwargs = reduce_kwargs or {}
        self.levels = levels
        self.futures = futures
        self._tag = None

        # Prepare storage for futures that have not been fully reduced.
        self._reducing: list[list[distributed.Future[T]]] = []
        for i in range(self.levels):
            self._reducing.append([])

    @property
    def tag(self) -> Any:
        """Tag associated with the data being reduced currently.

        The current tag is passed to the output function. It can be used as metadata for
        the current calculations.

        Note that changing the tag will result in an exception if there are pending
        reductions. It is recommended that you call the `flush` method prior to changing
        the tag.

        """
        return self._tag

    @tag.setter
    def tag(self, value: Any):
        for pending in self._reducing:
            if pending:
                raise RuntimeError(
                    _("cannot change tag when DaskReductionTree is not empty")
                )
        self._tag = value

    def add_futures(
        self, *futures: distributed.Future[T]
    ) -> list[distributed.Future[T]]:
        """Add futures to the reduction tree.

        Parameters
        ----------
        *futures
            Futures to reduce.

        Returns
        -------
        list[distributed.Future]
            Any futures submitted to the cluster to perform the reduction.

        """
        added = []

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
                    *self.reduce_args,
                    key=f"reduction-{level}-{tokenize(to_reduce)}",
                    **self.reduce_kwargs,
                )
                added.append(reduced)
                del self._reducing[level][: self.futures]

                # Output or add to the next level.
                if level == self.levels - 1:
                    self.output_func(reduced, self._tag)
                else:
                    self._reducing[level + 1].append(reduced)

        return added

    def flush(self) -> list[distributed.Future[T]]:
        """Flush the reduction tree.

        In most situations, the number of futures to reduce won't be an exact multiple
        of the reduction factor. Calling this method will reduce any futures remaining
        in the tree. There are three cases:

        * If the tree is empty, do nothing.
        * If there is one future in the entire tree, output that.
        * If there are multiple futures in the tree, call the reduction function on all
          of them and output the result.

        After this method has been called, the tree is guaranteed to be empty.

        Returns
        -------
        list[distributed.Future]
            Any futures submitted to the cluster to perform the reduction.

        """
        # Collect all remaining futures at any level.
        remaining = []
        for level in range(self.levels):
            remaining.extend(self._reducing[level])
            self._reducing[level].clear()

        if not remaining:
            return []

        if len(remaining) == 1:
            self.output_func(remaining[0], self._tag)
            return []

        # Schedule the reduction and output the future.
        reduced = self.client.submit(
            self.reduce_func,
            remaining,
            *self.reduce_args,
            key=f"reduction-finish-{tokenize(remaining)}",
            **self.reduce_kwargs,
        )
        self.output_func(reduced, self._tag)
        return [reduced]

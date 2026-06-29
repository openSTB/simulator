# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Base classes to help with controller implementation."""

from collections.abc import Iterable, Iterator
import itertools
import logging
from typing import Callable

from dask.tokenize import tokenize
import distributed
import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc
from openstb.simulator.utils.reduction import DaskReductionTree

_ = translations.load("openstb.simulator").gettext


class LoopController[T, P, R, C: abc.ControllerConfig](abc.Controller[C]):
    """Base class for a controller which loops over a set of pings, receivers etc.

    This loops over all combinations of a set of variables and performs a simulation for
    each of these. The inner simulations are performed in chunks over a set of targets
    and combined with a reduction tree. The chunks are submitted to a Dask cluster in a
    manner designed to keep the number of pending tasks within a certain threshold, and
    the tasks are monitored for failure.

    The inheriting plugin must set all `loop_*` attributes with no default value, and
    change those with default values if desired. It can then call the `loop_run` method
    to perform the simulation loop.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_futures = set()
        self.loop_sim_tasks = 0

    loop_sim_tasks: int
    """The number of simulation tasks currently submitted to the scheduler.

    This does not include non-simulation tasks such as reduction or result storage. This
    is intended as a status variable, and should not be modified by the inheriting
    plugin.

    """

    # This is not typed at distributed.Future[R] as the inheriting plugin may add other
    # futures with different result types.
    loop_futures: set[distributed.Future]
    """Pending futures for the current loop.

    The simulation and reduction tree futures are added internally. If the inheriting
    plugin adds other tasks to the cluster, for example to store results, it should add
    them to this set so they are managed by the loop.

    """

    loop_chunks: Callable[[tuple], Iterator[T]]
    """Chunks of targets to simulate.

    This must be set by the plugin class inheriting this class. It must be a callable
    which takes the current loop key and returns an iterator over the chunks of targets
    to simulate.

    """

    loop_params: Callable[[tuple], P]
    """Simulation parameters to use.

    This must be set by the plugin class inheriting this class. It must be a callable
    which takes the current loop key and returns the simulation parameters to use.

    """

    loop_simulate: Callable[[T, P], R]
    """Function to simulate one chunk of targets.

    This must be sent by the plugin class inheriting this class. It must be a callable
    which takes the target chunk and simulations parameters and returns the result.

    """

    loop_lower_threshold: float = 2.0
    """Lower threshold for number of simulation tasks in the scheduler.

    This is given in number of simulation tasks per worker. When the number of tasks in
    the scheduler drops below this, more are added.

    """

    loop_upper_threshold: float = 3.0
    """Upper threshold for number of simulation tasks in the scheduler.

    This is given in the number of simulation tasks per worker. When adding tasks as
    required by `loop_lower_threshold`, enough tasks will be added to reach this upper
    threshold.

    """

    loop_log_message: str = ""
    """Log message to output at the start of each loop.

    If set, this message is passed to a logger at an info level at the start of each
    loop. The loop key tuple will be unpacked and given to the logger so that the values
    can be included in the message.

    """

    @property
    def loop_running(self) -> bool:
        return len(self.loop_futures) > 0

    def loop_run(
        self,
        client: distributed.Client,
        loop_vars: Iterable[int | ArrayLike],
        rtree: DaskReductionTree[R],
    ):
        """Run the simulation loop.

        Parameters
        ----------
        client
            The Dask client to submit tasks to the cluster with.
        loop_vars
            Variables to loop over. Each may be a one-dimensional array of values, or an
            integer indicating the number of values N in which case the values will go
            from 0 to N - 1. Each combination of variables will be simulated. The loops
            will be performed in the given order, i.e., the rightmost variable will be
            used in the innermost loop.
        rtree
            The reduction tree to pass the outputs of the simulation function to.

        """
        if self.loop_running:
            raise RuntimeError("previously simulation loop is not complete")

        # Normalise the loop variables.
        loop_var_list: list[np.ndarray] = []
        for var in loop_vars:
            if isinstance(var, int):
                loop_var_list.append(np.arange(var))
            else:
                var_1d = np.atleast_1d(var).copy()
                if var_1d.ndim != 1:
                    raise ValueError(_("loop variables must be one-dimensional"))
                loop_var_list.append(var_1d)

        # Loop over the combinations of variables.
        for loop_key in itertools.product(*loop_var_list):
            try:
                self._loop_single(client, loop_key, rtree)
            except:
                # Cancel any other futures.
                for future in self.loop_futures:
                    future.cancel()
                self.loop_futures.clear()
                self.loop_sim_tasks = 0
                raise

        # Wait until the final tasks are complete.
        while self.loop_futures:
            try:
                self._loop_wait()
            except:
                # Cancel any other futures.
                for future in self.loop_futures:
                    future.cancel()
                self.loop_futures.clear()
                self.loop_sim_tasks = 0
                raise

    def _loop_single(
        self, client: distributed.Client, loop_key: tuple, rtree: DaskReductionTree[R]
    ):
        if self.loop_log_message:
            logger = logging.getLogger(__name__)
            logger.info(self.loop_log_message, *loop_key)

        # Get simulation inputs for this loop.
        chunks = self.loop_chunks(loop_key)
        params = self.loop_params(loop_key)

        # Determine the thresholds for this loop.
        N_workers = len(client.scheduler_info(n_workers=-1)["workers"])
        N_lower = int(np.ceil(N_workers * self.loop_lower_threshold))
        N_upper = int(np.ceil(N_workers * self.loop_upper_threshold))

        # Set the tag of the reduction tree so the output function can use it.
        rtree.tag = loop_key

        while True:
            # If we're below the lower threshold, add enough tasks to reach the upper.
            if self.loop_sim_tasks < N_lower:
                while self.loop_sim_tasks < N_upper:
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        self.loop_futures.update(rtree.flush())
                        return

                    future = client.submit(
                        self.loop_simulate,
                        chunk,
                        params,
                        key=f"simulate-{tokenize(chunk, loop_key)}",
                    )
                    self.loop_futures.add(future)
                    self.loop_sim_tasks += 1
                    self.loop_futures.update(rtree.add_futures(future))

            # Wait for something to complete.
            self._loop_wait()

    def _loop_wait(self) -> None:
        # Wait for a future to complete. Note that there may be more than one complete
        # in the same check.
        res = distributed.wait(self.loop_futures, return_when="FIRST_COMPLETED")
        self.loop_futures = res.not_done

        # Check each completed future for failures and update counts.
        for future in res.done:
            if future.key.startswith("simulate-"):
                self.loop_sim_tasks -= 1

            if future.status == "error":
                raise future.exception()
            if future.status == "cancelled":
                raise RuntimeError(_("a future was cancelled"))

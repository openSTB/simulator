# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Iterable

import distributed
import numpy as np
from numpy.typing import ArrayLike
import pytest

from openstb.simulator.controller import base
from openstb.simulator.plugin import abc
from openstb.simulator.utils.reduction import DaskReductionTree


# Implement abstract methods.
class DummyLoopController(base.LoopController):
    client: distributed.Client
    loop_vars: Iterable[int | ArrayLike]
    rtree: DaskReductionTree

    def config_class(self):
        return None

    def run(self, config):
        self.loop_run(self.client, self.loop_vars, self.rtree)


@pytest.mark.parametrize("where", ["loop", "end"])
def test_controller_base_loop_exception(test_cluster: abc.DaskCluster, where: str):
    """controller.LoopController: handles exceptions in simulation function"""

    def simulate(chunk, params):
        if chunk == 1:
            raise ValueError("chunk value not supported")
        return params[0]

    def store(result, loop_key):
        pass

    ctl = DummyLoopController()
    ctl.client = test_cluster.client
    ctl.rtree = DaskReductionTree(test_cluster.client, store, np.sum)
    ctl.loop_simulate = simulate
    ctl.loop_params = lambda k: k
    ctl.loop_vars = (1,)

    # If the number of tasks will be below the lower threshold, they will all be
    # submitted before _loop_single calls _loop_wait, and so the exception will be
    # captured by the 'wait for remaining tasks' block. If there are more than the
    # upper threshold, it will be captured by the 'loop iver variables' block.
    N_workers = len(test_cluster.client.scheduler_info(n_workers=-1)["workers"])
    if where == "end":
        ctl.loop_chunks = lambda k: iter(range(N_workers))
    else:
        ctl.loop_chunks = lambda k: iter(range(N_workers * 5))

    # Should raise the exception from the failing task.
    with pytest.raises(ValueError, match="chunk value not supported"):
        ctl.run({})

    # Should have been reset.
    assert not ctl.loop_futures
    assert ctl.loop_sim_tasks == 0


def test_controller_base_loop_already_running(test_cluster: abc.DaskCluster):
    """controller.LoopController: fails if simulation is already running"""

    def simulate(chunk, params):
        return params[0]

    def store(result, loop_key):
        pass

    ctl = DummyLoopController()
    ctl.client = test_cluster.client
    ctl.rtree = DaskReductionTree(test_cluster.client, store, np.sum)
    ctl.loop_simulate = simulate
    ctl.loop_chunks = lambda k: iter(range(5))
    ctl.loop_params = lambda k: k
    ctl.loop_vars = (1,)

    ctl.loop_futures = {1, 2, 3}  # type:ignore[arg-type]
    with pytest.raises(RuntimeError, match="previous.+loop.+not complete"):
        ctl.run({})


def test_controller_base_loop_multidim(test_cluster: abc.DaskCluster):
    """controller.LoopController: fails if loop_vars multi-dimensional"""

    def simulate(chunk, params):
        return params[0]

    def store(result, loop_key):
        pass

    ctl = DummyLoopController()
    ctl.client = test_cluster.client
    ctl.rtree = DaskReductionTree(test_cluster.client, store, np.sum)
    ctl.loop_simulate = simulate
    ctl.loop_chunks = lambda k: iter(range(5))
    ctl.loop_params = lambda k: k
    ctl.loop_vars = (np.ones((10, 10)),)

    with pytest.raises(ValueError, match="loop variables must be one-dimensional"):
        ctl.run({})

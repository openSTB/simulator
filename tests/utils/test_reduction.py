# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from dask import distributed
import numpy as np
import pytest

from openstb.simulator.plugin import abc
from openstb.simulator.utils import reduction


def test_utils_reduction_dask_basic(test_cluster: abc.DaskCluster):
    """utils.reduction.DaskReductionTree: basic operation"""
    reduced = []

    def output_func(future: distributed.Future, tag: tuple):
        nonlocal reduced
        reduced.append((future, tag))

    rtree = reduction.DaskReductionTree(
        test_cluster.client,
        output_func,
        np.sum,
        reduce_kwargs={"axis": 0},
        levels=2,
        futures=2,
    )
    rtree.tag = 22

    # Insert some futures singly.
    for i in range(3):
        arr = np.ones((100, 30), dtype=int)
        future = test_cluster.client.scatter(arr)
        rtree.add_futures(future)

    # And some more in one call.
    to_reduce = []
    for i in range(5):
        arr = np.ones((100, 30), dtype=int)
        to_reduce.append(test_cluster.client.scatter(arr))
    rtree.add_futures(*to_reduce)

    # Should have completed exactly two sets of trees with known values.
    assert len(reduced) == 2
    assert reduced[0][1] == 22
    res = reduced[0][0].result(timeout=1)
    assert res.shape == (100, 30)
    assert np.all(res == 4)
    assert reduced[1][1] == 22
    res = reduced[1][0].result(timeout=1)
    assert res.shape == (100, 30)
    assert np.all(res == 4)

    # Flushing should be a no-op at this stage.
    rtree.flush()
    assert len(reduced) == 2

    # If we add one more, it will not result in an output. Flushing should then just
    # output the original future.
    rtree.tag = ("a", "b")
    arr = np.ones((100, 30), dtype=int)
    future = test_cluster.client.scatter(arr)
    rtree.add_futures(future)
    assert len(reduced) == 2
    rtree.flush()
    assert len(reduced) == 3
    assert reduced[-1][0] is future
    assert reduced[-1][1] == ("a", "b")

    # Add more than one, but not enough to fully reduce. Flushing should reduce whatever
    # is left and output that.
    rtree.tag = 17
    for i in range(3):
        arr = np.ones((30, 100), dtype=int)
        future = test_cluster.client.scatter(arr)
        rtree.add_futures(future)

    assert len(reduced) == 3
    rtree.flush()
    assert len(reduced) == 4
    assert reduced[3][1] == 17
    res = reduced[3][0].result(timeout=1)
    assert res.shape == (30, 100)
    assert np.all(res == 3)


def test_utils_reduction_dask_tag(test_cluster: abc.DaskCluster):
    """utils.reduction.DaskReductionTree: check tag property"""

    def output_func(future: distributed.Future, tag: tuple):
        pass

    rtree = reduction.DaskReductionTree(test_cluster.client, output_func, np.sum)
    assert rtree.tag is None

    rtree.tag = 42
    assert rtree.tag == 42

    arr = np.ones((100, 30), dtype=int)
    future = test_cluster.client.scatter(arr)
    rtree.add_futures(future)

    with pytest.raises(RuntimeError, match="DaskReductionTree is not empty"):
        rtree.tag = 17

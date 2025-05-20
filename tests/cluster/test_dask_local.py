# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import logging
from math import floor

from dask.distributed import wait
import dask.system
import distributed.system
import pytest
import requests

from openstb.simulator.cluster.dask_local import DaskLocalCluster


@pytest.mark.cluster
def test_cluster_dask_local():
    """cluster: basic DaskLocalCluster operation"""
    c = DaskLocalCluster(workers=2, total_memory=0.01, dashboard_address=None)

    c.initialise()
    client = c.client
    assert client.status == "running"

    def workerfunc(num):
        from dask.distributed import get_worker

        return num, get_worker().id

    futures = c.client.map(workerfunc, [0, 1, 2, 3, 4, 5])
    wait(futures)
    results = [future.result() for future in futures]
    c.terminate()

    assert client.status == "closed"
    assert {r[0] for r in results} == {0, 1, 2, 3, 4, 5}
    assert len({r[1] for r in results}) == 2

    # Terminating twice should cause no issues.
    c.terminate()


@pytest.mark.cluster
def test_cluster_dask_local_dashboard(caplog):
    """cluster: DaskLocalCluster dashboard setting"""
    caplog.set_level(logging.INFO)

    # ":0" -> use a random available port.
    c = DaskLocalCluster(workers=2, total_memory=0.01, dashboard_address=":0")
    c.initialise()

    # Make sure initialising twice does not error.
    c.initialise()

    # Look through the logs to find the reported address.
    dashboard = None
    for record in caplog.records:
        if record.name != "distributed.scheduler":
            continue
        if "dashboard at:" in record.message:
            _, dashboard = record.message.rsplit(" ", 1)

    assert dashboard is not None, "dashboard address not reported"

    # Attempt to access the status page.
    response = requests.get(dashboard)
    assert response.status_code == 200


def test_cluster_dask_local_workers():
    """cluster: DaskLocalCluster settings for number of workers"""
    # Set number.
    c = DaskLocalCluster(workers=4, total_memory=0.01)
    assert c.workers == 4

    # All available.
    c = DaskLocalCluster(workers=-1, total_memory=0.01)
    assert c.workers == dask.system.CPU_COUNT

    # Fraction of available.
    c = DaskLocalCluster(workers=0.5, total_memory=0.01)
    assert c.workers == dask.system.CPU_COUNT // 2


def test_cluster_dask_local_memory():
    """cluster: DaskLocalCluster settings for memory usage"""
    # N.B., the DaskLocalCluster initialiser converts to an integer number of bytes per
    # worker, which is how the underlying distributed class expects it to be specified.

    # Number of bytes per worker.
    c = DaskLocalCluster(workers=2, worker_memory=1_000_000)
    assert c.memory == 1_000_000

    # Total number of bytes over all workers.
    c = DaskLocalCluster(workers=2, total_memory=1_000_000)
    assert c.memory == 500_000

    # Fraction of system memory per worker.
    c = DaskLocalCluster(workers=2, worker_memory=0.01)
    assert c.memory == int(floor(distributed.system.MEMORY_LIMIT * 0.01))

    # Fraction of system memory over all workers.
    c = DaskLocalCluster(workers=2, total_memory=0.01)
    assert c.memory == int(floor(distributed.system.MEMORY_LIMIT * 0.005))


def test_cluster_dask_local_error():
    """cluster: DaskLocalCluster error handling"""
    with pytest.raises(ValueError, match="worker_memory or total_memory must"):
        DaskLocalCluster(workers=2)

    with pytest.raises(ValueError, match="only one of worker_memory and total_memory"):
        DaskLocalCluster(workers=2, worker_memory=0.1, total_memory=0.2)

    c = DaskLocalCluster(workers=2, total_memory=0.1)
    with pytest.raises(RuntimeError, match="must initialise.+before"):
        c.client

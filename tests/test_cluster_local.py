# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import logging
from math import floor
import socket

from dask.distributed import wait
import dask.system
import distributed.system
import pytest

from openstb.simulator import cluster


@pytest.mark.cluster
def test_cluster_local():
    """cluster: basic LocalCluster operation"""
    c = cluster.LocalCluster(workers=2, total_memory=0.01, dashboard_address=None)

    client = c.initialise()
    assert client.status == "running"

    def workerfunc(num):
        from dask.distributed import get_worker

        return num, get_worker().id

    futures = client.map(workerfunc, [0, 1, 2, 3, 4, 5])
    wait(futures)
    results = [future.result() for future in futures]
    c.terminate()

    assert client.status == "closed"
    assert {r[0] for r in results} == {0, 1, 2, 3, 4, 5}
    assert len({r[1] for r in results}) == 2


@pytest.mark.cluster
def test_cluster_local_dashboard(caplog):
    """cluster: LocalCluster dashboard setting"""
    caplog.set_level(logging.INFO)

    # ":0" -> use a random available port.
    with cluster.LocalCluster(workers=2, total_memory=0.01, dashboard_address=":0"):
        # Look through the logs to find the reported address.
        d_addr = None
        d_port = None
        d_page = None
        for record in caplog.records:
            if record.name != "distributed.scheduler":
                continue
            if "dashboard at:" in record.message:
                _, _, url = record.message.partition("//")
                d_addr, _, port_page = url.partition(":")
                port, _, d_page = port_page.partition("/")
                d_port = int(port)
                break

        assert d_addr is not None, "dashboard address not reported"
        assert d_port is not None, "dashboard port not reported"
        assert d_page is not None, "dashboard page not reported"

        # Attempt to access the status page.
        sock = socket.socket()
        sock.settimeout(1.0)
        sock.connect((d_addr, d_port))
        sock.send(f"GET /{d_page} HTTP/1.1\n\n".encode("ascii"))
        response = sock.recv(15)
        sock.close()
        assert response == b"HTTP/1.1 200 OK", "could not access dashboard"


def test_cluster_local_workers():
    """cluster: LocalCluster settings for number of workers"""
    # Set number.
    c = cluster.LocalCluster(workers=4, total_memory=0.01)
    assert c.workers == 4

    # All available.
    c = cluster.LocalCluster(workers=-1, total_memory=0.01)
    assert c.workers == dask.system.CPU_COUNT

    # Fraction of available.
    c = cluster.LocalCluster(workers=0.5, total_memory=0.01)
    assert c.workers == dask.system.CPU_COUNT // 2


def test_cluster_local_memory():
    """cluster: LocalCluster settings for memory usage"""
    # N.B., the LocalCluster initialiser converts to an integer number of bytes per
    # worker, which is how the underlying distributed class expects it to be specified.

    # Number of bytes per worker.
    c = cluster.LocalCluster(workers=2, worker_memory=1_000_000)
    assert c.memory == 1_000_000

    # Total number of bytes over all workers.
    c = cluster.LocalCluster(workers=2, total_memory=1_000_000)
    assert c.memory == 500_000

    # Fraction of system memory per worker.
    c = cluster.LocalCluster(workers=2, worker_memory=0.01)
    assert c.memory == int(floor(distributed.system.MEMORY_LIMIT * 0.01))

    # Fraction of system memory over all workers.
    c = cluster.LocalCluster(workers=2, total_memory=0.01)
    assert c.memory == int(floor(distributed.system.MEMORY_LIMIT * 0.005))


def test_cluster_local_error():
    """cluster: LocalCluster error handling"""
    with pytest.raises(ValueError, match="worker_memory or total_memory must"):
        cluster.LocalCluster(workers=2)

    with pytest.raises(ValueError, match="only one of worker_memory and total_memory"):
        cluster.LocalCluster(workers=2, worker_memory=0.1, total_memory=0.2)

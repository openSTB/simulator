# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from typing import Generator

from distributed.client import Client
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.utils_test import cluster
from distributed.worker import Worker
import pytest

from openstb.simulator.plugin.abc import DaskCluster


@pytest.fixture(scope="session")
def _session_cluster() -> Generator[tuple[dict, list[dict]]]:
    """Session-scoped cluster to use for testing."""

    class WorkerCoverage(WorkerPlugin):
        """Plugin to enable coverage in worker processes.

        This has to be defined here; if defined at the top level of the module, then
        Dask will try to serialize it by name which cannot be found in the other
        processes ("no such module: tests").

        """

        def setup(self, worker: Worker):
            import os

            import coverage

            # Start coverage for this process using the project configuration.
            os.environ["COVERAGE_PROCESS_START"] = "pyproject.toml"
            self.cov = coverage.process_startup()

        def teardown(self, worker: Worker):
            if self.cov is not None:
                self.cov.save()

    with cluster() as (scheduler, workers):
        # Register the plugin to start coverage on all workers.
        with Client(scheduler["address"]) as client:
            client.register_plugin(WorkerCoverage(), name="coverage")

        yield (scheduler, workers)

        # Worker processes are killed during cluster shutdown before coverage can be
        # written. Unregistering the plugin calls its teardown() and saves it.
        with Client(scheduler["address"]) as client:
            client.unregister_worker_plugin("coverage")


class TestFixtureCluster(DaskCluster):
    """DaskCluster plugin wrapping an existing client."""

    def __init__(self, client: Client):
        self._client = client

    def initialise(self):
        pass

    @property
    def client(self) -> Client:
        return self._client


@pytest.fixture(scope="function")
def test_cluster(_session_cluster: tuple[dict, list[dict]]) -> Generator[DaskCluster]:
    """Function-scoped fixture return a test abc.DaskCluster"""
    # Connect a client to the session-wide cluster.
    scheduler, workers = _session_cluster
    with Client(scheduler["address"]) as client:
        try:
            # And give the test a DaskCluster wrapper to it.
            yield TestFixtureCluster(client)
        finally:
            # Futures should drop out of scope during teardown, but lets be sure.
            for future in client.futures.values():
                future.cancel()

            client.close()

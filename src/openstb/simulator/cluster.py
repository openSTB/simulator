# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Cluster interfaces for openSTB simulator calculations."""

from dask.system import CPU_COUNT
import distributed
from distributed.system import MEMORY_LIMIT
import numpy as np

from openstb.i18n.support import domain_translator
from openstb.simulator.plugin.abc import DaskCluster


_ = domain_translator("openstb.simulator", plural=False)


class DaskLocalCluster(DaskCluster):
    """A Dask cluster running on the local machine."""

    def __init__(
        self,
        workers: int | float,
        worker_memory: int | float | None = None,
        total_memory: int | float | None = None,
        security: bool = True,
        dashboard_address: str | None = None,
    ):
        """
        Parameters
        ----------
        workers : int, float
            Number of workers to add to the cluster. If a float, this is interpreted as
            a fraction of the available CPUs. If -1, use all available CPUs. Any other
            value is taken to be the number of CPUs to use.
        worker_memory, total_memory : int, float
            The desired memory limit for the cluster. This can be specified per worker
            or for all workers; only one may be given (and one must be given). A float
            indicates a fraction of the total system memory, and an integer the number
            of bytes. Note that this limit is enforced on a best-effort basis and some
            workers may exceed it.
        security : boolean
            If True, self-signed temporary credentials are used to secure communications
            within the cluster. If False, communications are unencrypted.
        dashboard_address : str, optional
            Address the Dask diagnostic dashboard server will listen on, e.g.,
            "localhost:8787" or "0.0.0.0:8787". If not given, the server will be
            disabled. Note that the logs may still print a dashboard address if
            disabled, but there will be nothing at that address. This is a known bug in
            dask.distributed: https://github.com/dask/distributed/issues/7994

        """
        if isinstance(workers, float):
            self.workers = int(np.fix(workers * CPU_COUNT))
        elif workers == -1:
            self.workers = CPU_COUNT
        else:
            self.workers = workers

        if worker_memory is None and total_memory is None:
            raise ValueError(_("worker_memory or total_memory must be given"))
        if worker_memory is not None and total_memory is not None:
            raise ValueError(
                _("only one of worker_memory and total_memory can be given")
            )

        # The distributed LocalCluster takes memory per worker.
        if worker_memory is not None:
            if isinstance(worker_memory, float):
                worker_memory = int(np.fix(worker_memory * MEMORY_LIMIT))
            self.memory = worker_memory
        elif total_memory is not None:
            if isinstance(total_memory, float):
                total_memory = int(np.fix(total_memory * MEMORY_LIMIT))
            self.memory = total_memory // self.workers

        self.security = security
        self.dashboard_address = dashboard_address

    def initialise(self) -> distributed.Client:
        self._cluster = distributed.LocalCluster(
            n_workers=self.workers,
            processes=True,
            threads_per_worker=1,
            memory_limit=self.memory,
            dashboard_address=self.dashboard_address,
            security=self.security,
        )
        self._client = self._cluster.get_client()
        return self._client

    def terminate(self):
        self._client.shutdown()
        self._client = None
        self._cluster.close()
        self._cluster = None


class DaskMPICluster(DaskCluster):
    """A cluster of Dask nodes communicating via MPI.

    This requires the ``dask_mpi`` package to be installed. This uses the ``mpi4py``
    library to communicate via MPI. The process running with MPI rank 0 is used for the
    Dask scheduler, and the process running with MPI rank 1 is used for the main
    simulation controller. All other processes are used as computation workers.

    """

    def __init__(
        self, interface: str | None = None, dashboard_address: str | None = None
    ):
        """
        Parameters
        ----------
        interface : str
            The network interface to use for communication, e.g., "eth0" or "ib0". If
            not specified, the scheduler will attempt to determine the appropriate
            interface.
        dashboard_address : str, optional
            Address the Dask diagnostic dashboard server will listen on, e.g.,
            "localhost:8787" or "0.0.0.0:8787". If not given, the server will be
            disabled.

        """
        # Check now that MPI is available on each worker.
        try:
            import dask_mpi  # noqa:F401
        except (ImportError, ModuleNotFoundError):  # pragma:no cover
            raise ValueError(
                _("the dask_mpi package must be installed to use an MPI cluster")
            )

        self.interface = interface
        self.dashboard_address = dashboard_address

    def initialise(self) -> distributed.Client:
        import dask_mpi

        dask_mpi.initialize(
            interface=self.interface,
            dashboard=self.dashboard_address is not None,
            dashboard_address=self.dashboard_address or "",
        )

        return distributed.Client()

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import logging

import dask_mpi
import distributed
from mpi4py import MPI

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import DaskCluster

_ = translations.load("openstb.simulator").gettext


class DaskMPICluster(DaskCluster):
    """A cluster of Dask nodes communicating via MPI.

    This requires the ``dask_mpi`` package to be installed. This uses the ``mpi4py``
    library to communicate via MPI. The process running with MPI rank 0 is used for the
    Dask scheduler, and the process running with MPI rank 1 is used for the main
    simulation controller. All other processes are used as computation workers.

    """

    def __init__(
        self,
        interface: str | None = None,
        dashboard_address: str | None = None,
        separate_workers: bool = True,
        local_directory: str | None = None,
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
        separate_workers : boolean, default True
            If True, the worker processes (all processes with a rank other than 1) will
            use the initialise_workers() method. If False, all processes will read the
            configuration and proceed as normal. Setting this to True is recommended so
            that only the main controller process will have to read and parse the
            configuration.
        local_directory : str, optional
            The path to a local scratch directory for Dask to use. This should be local
            to each node, not on a network drive. If not given, Dask will fall back to
            an internal default path.

        """
        self.interface = interface
        self.dashboard_address = dashboard_address
        self._initialised = False
        self._client: distributed.Client | None = None
        self.separate_workers = separate_workers
        self.local_directory = local_directory

    def initialise(self):
        if self._initialised:
            return

        # Settings for dask_mpi.initialize().
        settings = dict(
            interface=self.interface,
            dashboard=self.dashboard_address is not None,
            dashboard_address=self.dashboard_address or "",
            local_directory=self.local_directory,
        )

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if self.separate_workers:
            # Avoid a misconfiguration with a clear error message.
            if rank != 1:
                raise RuntimeError(
                    _(
                        "when using separate workers, the simulation controller should "
                        "only be run on MPI rank 1"
                    )
                )

            # Broadcast the settings to the scheduler and worker processes. Note that
            # MPI inserts a synchronisation barrier with a broadcast.
            comm.bcast(settings, root=1)

        if rank == 1:
            logger = logging.getLogger(__name__)
            logger.info(
                "Initialising Dask MPI cluster on rank %(rank)d of %(size)d",
                {"rank": rank, "size": comm.Get_size()},
            )

        dask_mpi.initialize(**settings)
        self._initialised = True

        if rank == 1 and self.dashboard_address is not None:
            if self.dashboard_address.startswith(":"):
                url = f"http://127.0.0.1{self.dashboard_address}"
            else:
                url = f"http://{self.dashboard_address}"
            logger.info(_("Dask dashboard is at %(url)s"), {"url": url})

    @classmethod
    def initialise_worker(cls):
        comm = MPI.COMM_WORLD

        # Avoid misconfigurations with a clear error message.
        rank = comm.Get_rank()
        if rank == 1:
            return True

        # The controller will broadcast the settings when available (after the
        # configuration has been loaded and parsed, and the main DaskCluster plugin is
        # initialised).
        settings = comm.bcast(None, root=1)
        dask_mpi.initialize(**settings)
        return False

    @property
    def client(self) -> distributed.Client:
        if not self._initialised:
            raise RuntimeError(_("must initialise the cluster before getting a client"))
        if self._client is None:
            self._client = distributed.Client()
        return self._client

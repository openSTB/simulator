# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import logging

from dask.system import CPU_COUNT
import distributed
from distributed.system import MEMORY_LIMIT
import numpy as np

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import DaskCluster

_ = translations.load("openstb.simulator").gettext


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

        # The distributed LocalCluster takes memory per worker as an integer number of
        # bytes. Convert our inputs.
        if worker_memory is not None and total_memory is not None:
            raise ValueError(
                _("only one of worker_memory and total_memory can be given")
            )
        elif worker_memory is not None:
            if isinstance(worker_memory, float):
                worker_memory = int(np.fix(worker_memory * MEMORY_LIMIT))
            self.memory = worker_memory
        elif total_memory is not None:
            if isinstance(total_memory, float):
                total_memory = int(np.fix(total_memory * MEMORY_LIMIT))
            self.memory = total_memory // self.workers
        else:
            raise ValueError(_("worker_memory or total_memory must be given"))

        self.security = security
        self.dashboard_address = dashboard_address

        self._cluster: distributed.LocalCluster | None = None
        self._client: distributed.Client | None = None

    def initialise(self):
        if self._cluster is not None:
            return

        logger = logging.getLogger(__name__)
        logger.info(
            _(
                "Initialising local Dask cluster with %(N)d workers "
                "(%(ram_MiB)dMiB RAM per worker)"
            ),
            {"N": self.workers, "ram_MiB": self.memory / (1024 * 1024)},
        )

        self._cluster = distributed.LocalCluster(
            n_workers=self.workers,
            processes=True,
            threads_per_worker=1,
            memory_limit=self.memory,
            dashboard_address=self.dashboard_address,
            security=self.security,
        )

        if self.dashboard_address is not None:
            if self.dashboard_address.startswith(":"):
                url = f"http://127.0.0.1{self.dashboard_address}"
            else:
                url = f"http://{self.dashboard_address}"
            logger.info(_("Dask dashboard is at %(url)s"), {"url": url})

    @property
    def client(self) -> distributed.Client:
        if self._cluster is None:
            raise RuntimeError(_("must initialise the cluster before getting a client"))

        if self._client is None:
            self._client = self._cluster.get_client()
        return self._client

    def terminate(self):
        logger = logging.getLogger(__name__)
        logger.info(_("Shutting down local Dask cluster"))
        if self._client is not None:
            self._client.shutdown()
            self._client = None
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None

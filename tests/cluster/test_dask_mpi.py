# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import ast
import os
import subprocess
import sys

import pytest

# mpi4py should be installed as a dependency of dask_mpi, but let's make sure since our
# tests directly use mpi4py.
pytest.importorskip("dask_mpi", reason="Need dask_mpi for DaskMPICluster")
pytest.importorskip("mpi4py", reason="Need mpi4py for tests")

from openstb.simulator.cluster.dask_mpi import DaskMPICluster  # noqa: E402

# We also need some way of starting an MPI job.
try:
    subprocess.run(["mpirun", "-V"], capture_output=True, check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    pytest.skip("mpirun not available", allow_module_level=True)


@pytest.mark.cluster
def test_cluster_dask_mpi(tmp_path):
    # Generate a test script which will start the cluster, run some very simple tasks
    # and print the resulting sets of data.
    script_fn = tmp_path / "test_script.py"
    script_fn.write_text(
        """
# Enable coverage in the process that will run this script.
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 1:
    import coverage
    coverage.process_startup()


from dask.distributed import wait
from openstb.simulator.cluster.dask_mpi import DaskMPICluster


def workerfunc(i):
    return i, MPI.COMM_WORLD.Get_rank()


# Initialise twice to ensure it handles this.
c = DaskMPICluster(separate_workers=False)
c.initialise()
c.initialise()

assert c.client.status == "running"

futures = c.client.map(workerfunc, [0, 1, 2, 3, 4, 5])
wait(futures)

results = [future.result() for future in futures]
jobs = {result[0] for result in results}
print(jobs)
ranks = {result[1] for result in results}
print(ranks)
"""
    )

    # Configure the environment; coverage.py needs to know its configuration file.
    env = dict(os.environ)
    env["COVERAGE_PROCESS_START"] = "pyproject.toml"

    # Run the test and read the result.
    result = subprocess.run(
        ["mpirun", "-n", "4", sys.executable, str(script_fn)],
        capture_output=True,
        check=True,
        text=True,
        env=env,
        timeout=15,
    )
    output = result.stdout.splitlines()

    # Should have six jobs, numbered zero through 5, printed in set notation.
    jobs = ast.literal_eval(output[0])
    assert jobs == {0, 1, 2, 3, 4, 5}

    # mpirun -n 4 will generate four workers. Worker rank 0 will be used for the Dask
    # scheduler and rank 1 to run the controlling script, leaving 2 and 3 for tasks.
    ranks = ast.literal_eval(output[1])
    assert ranks == {2, 3}


def test_cluster_dask_mpi_error():
    """DaskMPICluster error handling"""
    c = DaskMPICluster()
    with pytest.raises(RuntimeError, match="must initialise.+before"):
        c.client

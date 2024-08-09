<!--

SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent

-->

Simple point target simulation
==============================

The `simulate.py` script configures a simulation of some point targets. It does this by
creating a dictionary holding the various plugins needed for the simulation. This
configuration dictionary is then given to the `run()` method of the simulation
controller to perform the simulation.

If run directly, i.e., with `python simulate.py`, a local Dask cluster with 8 workers
will be used for the simulation. Alternatively, the `run_with_mpi.sh` shell script can
be run to execute the simulation within an MPI environment; in this case 6 workers will
be used in the cluster, with the other 2 used to run the simulation controller and
manage the Dask scheduler. In either case, the Dask diagnostic dashboard showing how the
cluster is being utilised will be available at http://127.0.0.1:8787/ for the duration
of the simulation.

The initial results of the simulation are stored in a [zarr](https://zarr.readthedocs.io/)
file. The simulation script configures a result converter plugin to convert this to a
NumPy file at `simple_points.npz`. It also includes a commented-out configuration to
convert the output to a MATLAB file if you prefer. See the `plot_results.py` script
for examples of how to load and plot the NumPy-formatted simulation results.

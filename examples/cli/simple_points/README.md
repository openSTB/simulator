<!--

SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent

-->

Simple point target simulation
==============================

The `simple_points.toml` file is a configuration file in the [TOML](https://toml.io/)
format defining the plugins to be used for a simulation. This contains the same
simulation configuration as used in the `scripts/simple_points` example. The
configuration file is commented with some details of its expected format.

To run the simulation, use the command `openstb-sim run simple_points.toml`. The CLI
should automatically select the correct loader to parse the configuration (like
everything else, the code to load configuration files is a plugin). You could also force
it to use a particular loader if the CLI fails to pick the correct one, in this case
`openstb-sim run -c toml simple_points.toml`. A local cluster with 8 workers is used for
the simulation. The Dask diagnostic dashboard showing how the cluster is being utilised
will be available at http://127.0.0.1:8787/ for the duration of the simulation.

The initial results of the simulation are stored in a [zarr](https://zarr.readthedocs.io/)
file. The configuration includes a result converter plugin to convert this to a NumPy
file at `simple_points.npz`. It also includes a commented-out configuration to convert
the output to a MATLAB file if you prefer. See the `plot_results.py` script for examples
of how to load and plot the NumPy-formatted simulation results.

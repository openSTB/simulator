<!--

SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent

-->

openSTB sonar simulation framework
==================================

The openSTB sonar simulation is a modular framework for simulating the signals received
by a sonar system. It is currently in the early stages of development, so bugs are to be
expected.


License
-------

The simulator is available under the BSD-2-Clause plus patent license, the text of which
can be found in the `LICENSES/` directory or
[online](https://spdx.org/licenses/BSD-2-Clause-Patent.html). Some of the supporting
files are under the Creative Commons Zero v1.0 Universal license (effectively public
domain). Again, the license is available in the `LICENSES/` directory or
[online](https://spdx.org/licenses/CC0-1.0.html).


Installation
------------

The openSTB tools are not yet published on PyPI (but will be in the near future). Before
installing the simulator, you will need to have the companion [internationalisation
support package](https://github.com/openSTB/i18n) installed in your environment, as well
as the [hatchling build backend](https://pypi.org/project/hatchling/). You can then run
`python -m pip install --no-build-isolation .` from the top level of a clone of this
repository (the `--no-build-isolation` prevents pip trying to build it in a clean
environment without access to the support packages). Some optional packages may also
need to be installed for your use case. If you want to use MPI-based parallelisation,
add the `MPI` option to the install: `python -m pip install --no-build-isolation
".[MPI]"`. To also install tools to help develop code for the framework, add the `dev`
option: `python -m pip install --no-build-isolation ".[dev]"`. You can give both options
separated by a comma, i.e., `python -m pip install --no-build-isolation ".[MPI,dev]"`.


Running the simulator
---------------------

At this stage, the simulator can only be used as a library; in the future, a simple CLI
is planned to run the simulator from a configuration file. See the `example_sim.py`
script for a simple example simulation. You can run this directly as `python
example_sim.py` to perform the simulation with a local cluster of 8 processes, or use
the `example_sim_mpi.sh` to execute it within an MPI environment. While the simulation
is running, a Dask diagnostic dashboard showing how the cluster is being utilised will
be available at http://127.0.0.1:8787/.

The initial output of the simulation is stored in a [zarr](https://zarr.readthedocs.io/)
file. The example simulation configures a result converter plugin to convert this to a
NumPy file at `example_sim.npz`. It also includes a commented-out configuration to
convert the output to a MATLAB file if you prefer. See the `example_sim_plot.py` script
for examples of how to load and plot the NumPy-formatted simulation results.

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Installation


## From PyPI

The simulator is published on [PyPI](https://pypi.org) as
[`openstb-simulator`](https://pypi.org/project/openstb-simulator). You can use your
preferred Python environment management tool to install it. For example, with pip:

```console
pip install openstb-simulator
```

## Optional dependencies

There are a number of sets of optional dependencies you may wish to add to the
installation:

* `dask-diagnostics`: support for Dask's [interactice diagnostic
  dashboard](https://docs.dask.org/en/stable/dashboard.html) to visualise performance.

* `mpi`: required for using the simulator on an MPI-based cluster

* `doc`: tools for building the documentation

* `tests`: tools for running the unit tests in the source repository

* `dev`: tools useful for helping to develop the simulator. This automatically includes
  the `dask-diagnotics`, `doc` and `tests` dependencies.

These options can be specified when installing the simulator. For example, if you use
pip to install from PyPI, you can just add your desired set of dependencies in square
brackets:

```console
pip install 'openstb-simulator[dask-diagnostics,mpi]'
```

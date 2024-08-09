#!/usr/bin/sh

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

# Ask MPI to run the simulation script with eight processes. One of these will be used
# by Dask for the scheduler, and another will manage the simulation. The other six will
# be used as workers. You will be able to access the Dask dashboard at
# http://127.0.0.1:8787 while the simulation is running.
mpirun -n 8 python simulate.py mpi

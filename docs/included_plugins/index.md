---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Other included plugins

The following sections list the plugins included with the simulator which are not yet
individually documented. The plugin name they can be loaded with is shown. The API
documentation of the plugin code is linked to for more details about the parameters for
each plugin.


## Configuration loader

* `toml`: read configuration from a [TOML](https://toml.io) file
  ([API documentation][openstb.simulator.config_loader.toml.TOMLLoader])


## Dask cluster

* `local`: use an ad-hoc cluster on the local machine
  ([API documentation][openstb.simulator.cluster.dask_local.DaskLocalCluster])

* `mpi`: use an HPC cluster communicating over MPI
  ([API documentation][openstb.simulator.cluster.dask_mpi.DaskMPICluster])


## Point targets

* `random_point_rectangle`: points randomly distributed inside a rectangular area
  ([API documentation][openstb.simulator.target.points.RandomPointRect])

* `random_point_triangle`: points randomly distributed inside a triangle
  ([API documentation][openstb.simulator.target.points.RandomPointTriangle])

* `single_point`: single point target at a specified location
  ([API documentation][openstb.simulator.target.points.SinglePoint])


## Signal windows

* `blackman`
  ([API documentation][openstb.simulator.system.signal_windows.BlackmanWindow])

* `blackman_harris`
  ([API documentation][openstb.simulator.system.signal_windows.BlackmanHarrisWindow])

* `generalised_cosine`
  ([API documentation][openstb.simulator.system.signal_windows.GeneralisedCosineWindow])

* `hamming`
  ([API documentation][openstb.simulator.system.signal_windows.HammingWindow])

* `hann`
  ([API documentation][openstb.simulator.system.signal_windows.HannWindow])

* `nuttall`
  ([API documentation][openstb.simulator.system.signal_windows.NuttallWindow])

* `tukey`
  ([API documentation][openstb.simulator.system.signal_windows.TukeyWindow])


## System description

* `generic`: basic collection of transducers and the transmitted signal
  ([API documentation][openstb.simulator.system.GenericSystem])


## Transducers

* `generic`
  ([API documentation][openstb.simulator.system.transducer.GenericTransducer])


## Travel time calculators

* `iterative`: use a iterative procedure to account for intra-ping motion
  ([API documentation][openstb.simulator.travel_time.iterative.Iterative])

* `stop_and_hop`: apply the stop-and-hop assumption when calculating travel time
  ([API documentation][openstb.simulator.travel_time.stop_and_hop.StopAndHop])

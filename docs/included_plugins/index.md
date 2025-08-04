---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Included plugins

The following sections list the plugins included with the simulator. The plugin name
they can be loaded with is shown. The API documentation of the plugin code is linked to
for more details about the parameters for each plugin. As this documentation is
extended, further details and examples of the plugins will be added.


## Configuration loader

* `toml`: read configuration from a [TOML](https://toml.io) file
  ([API documentation][openstb.simulator.config_loader.toml.TOMLLoader])


## Dask cluster

* `local`: use an ad-hoc cluster on the local machine
  ([API documentation][openstb.simulator.cluster.dask_local.DaskLocalCluster])

* `mpi`: use an HPC cluster communicating over MPI
  ([API documentation][openstb.simulator.cluster.dask_mpi.DaskMPICluster])


## Distortion

* `anslie_mccolm_attenuation`: acoustic attenuation using the model of Anslie and McColm
  ([API documentation][openstb.simulator.distortion.environmental.AnslieMcColmAttenuation])

* `doppler`: model the distortion of the spectrum due to the Doppler effect
  ([API documentation][openstb.simulator.distortion.doppler.DopplerDistortion])

* `geometric_spreading`: energy loss due to geometric spreadding
  ([API documentation][openstb.simulator.distortion.environmental.GeometricSpreading])

* `rectangular_beampattern`: beampattern of an ideal rectangular transducer
  ([API documentation][openstb.simulator.distortion.beampattern.RectangularBeampattern])


## Environment

* `invariant`: spatially and temporally invariant operating environment
  ([API documentation][openstb.simulator.environment.invariant.InvariantEnvironment])


## Ping times

* `constant_distance`: travel a fixed distance between pings
  ([API documentation][openstb.simulator.system.ping_times.ConstantDistance])

* `constant_interval`: emit pings at a constant rate
  ([API documentation][openstb.simulator.system.ping_times.ConstantInterval])


## Point targets

* `random_point_rectangle`: points randomly distributed inside a rectangular area
  ([API documentation][openstb.simulator.target.points.RandomPointRect])

* `single_point`: single point target at a specified location
  ([API documentation][openstb.simulator.target.points.SinglePoint])


## Result converters

* `matlab`: save results to a MATLAB data file
  ([API documentation][openstb.simulator.result_converter.matlab.MATLABConverter])

* `numpy`: save results to a NumPy `.npz` file
  ([API documentation][openstb.simulator.result_converter.numpy.NumpyConverter])


## Signals

* `lfm_chirp`: transmit a linear frequency modulated chirp
  ([API documentation][openstb.simulator.system.signal.LFMChirp])


## Simulation types

* `points`: simulate using idealised point targets
  ([API documentation][openstb.simulator.simulation.points.PointSimulation])


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


## Trajectory

* `linear`: idealised linear trajectory
  ([API documentation][openstb.simulator.system.trajectory.Linear])


## Transducers

* `generic`
  ([API documentation][openstb.simulator.system.transducer.GenericTransducer])


## Travel time calculators

* `iterative`: use a iterative procedure to account for intra-ping motion
  ([API documentation][openstb.simulator.travel_time.iterative.Iterative])

* `stop_and_hop`: apply the stop-and-hop assumption when calculating travel time
  ([API documentation][openstb.simulator.travel_time.stop_and_hop.StopAndHop])

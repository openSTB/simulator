---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Simple point simulation

For this example, we will configure a simple simulation of some point targets. The
first part of the configuration needs to specify the type of simulation to run. Here we
want a point target simulation. We will divide the scene into chunks of 500 targets
each, allowing us to split the work across multiple CPUs. We set the sampling rate and
frequency used for basebanding, and finally set a filename for the simulator to store
its results under.

```toml
[simulation]
plugin = "points"
result_filename = "simple_points.zarr"
targets_per_chunk = 500
sample_rate = 30e3
baseband_frequency = 110e3
```

We want to convert the results into a more familiar format for us to use. The simulator
will store the results in the file named above, but we can configure a result converter
plugin which is run at the end of the simulation to convert to our desired format (the
original results file the simulator used will then be removed). We could ask for the
results to be stored in a NumPy `.npz` file:

``` toml
[result_converter]
plugin = "numpy"
filename = "simple_points.npz"
compress = false
```

Or we could store them in a MATLAB file.

``` toml
[result_converter]
plugin = "matlab"
filename = "simple_points.mat"
```

We then need to tell the simulator to use an ad-hoc Dask cluster on our local computer.
Here we want to use a maximum of 8 workers and up to 40% of the total system memory
(note that the memory limit is best-effort, not strictly enforced). By default,
the Dask dashboard is not started, so we have to explicitly state an address it will
listen on -- in this case port 8787 on the local computer, i.e., http://127.0.0.1:8787.

```toml
[dask_cluster]
plugin = "local"
workers = 8
total_memory = 0.4
dashboard_address = ":8787"
```

Next, lets specify the details of the transmitter. The coordinate system is x forwards,
y starboard and z down.  The orientation is a quaternion defining how to rotate the
transducer from its original pose of pointing along the x axis. The value here results
in it pointing to starboard and 15 degrees below horizontal. We can also specify the
beampattern of the transducer via a signal distorting plugin.

```toml
[transmitter]
plugin = "generic"
position = [0, 1.2, 0.3]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]

[transmitter.beampattern]
plugin = "rectangular_beampattern"
width = 0.015
height = 0.03
transmit = true
receive = false
frequency = "centre"  # only calculate the scale factor at the centre frequency
```

The transmitter will also need a signal to send, in this case an LFM up-chirp.

```toml
[signal]
plugin = "lfm_chirp"
f_start = 100e3
f_stop = 120e3
duration = 0.015
rms_spl = 190
window = {name="tukey", alpha=0.2}
```

We want to use multiple receivers to form a linear array. These are specified with the
same options as the transmitter, but we use double square brackets around the header to
indicate there are multiple values (an *array of tables* in TOML terms). Channel 0 of
the results will correspond to the first receiver defined in the configuration file and
so forth.

Here we use a 5-element receiver array positioned above the transmitter (remember z is
down) and with the same orientation.

```toml
[[receivers]]
plugin = "generic"
position = [-0.1, 1.2, 0.0]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]

[[receivers]]
plugin = "generic"
position = [-0.05, 1.2, 0.0]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]

[[receivers]]
plugin = "generic"
position = [0, 1.2, 0.0]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]

[[receivers]]
plugin = "generic"
position = [0.05, 1.2, 0.0]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]

[[receivers]]
plugin = "generic"
position = [0.1, 1.2, 0.0]
orientation = [0.70105738, 0.09229596, -0.09229596, 0.70105738]
```

The simulator also needs to know the trajectory the system followed. For our purposes,
an ideal linear trajectory is sufficient.

```toml
[trajectory]
plugin = "linear"
start_position = [0, 0, 0]
end_position = [10, 0, 0]
speed = 1.5
```

Next we have to specify when the sonar emits pings. Here, we say it will ping every
0.2s, starting at the start of the trajectory and with the final ping being at least
0.5s before the end of the trajectory.

```toml
[ping_times]
plugin = "constant_interval"
interval = 0.2
start_delay = 0
end_delay = 0.5
```

Some environmental parameters are needed to perform the simulation. For simplicity, we
use an invariant (constant) environment with fixed values.

```toml
[environment]
plugin = "invariant"
salinity = 14.5
sound_speed = 1480
temperature = 11.2
```

Now lets define some targets to simulate. Like with the receivers, this is an array of
tables.

First, a rectangle with points randomly scattered within it to achieve a desired
density. Remember that z is down, so a normal with a negative z component means the
normal is pointing up. The reflectivity is the fraction of the incident amplitude that
is reflected back to the sonar.

```toml
[[targets]]
plugin = "random_point_rectangle"
seed = 10671
Dx = 5
Dy = 120
centre = [5, 75, 10]
normal = [0, 0, -1]
point_density = 10
reflectivity = 0.06
```

And then lets add a single target at a specific position with a much stronger echo.

```toml
[[targets]]
plugin = "single_point"
position = [5, 40, 10]
reflectivity = 10
```

One of the key parts of a simulation is to calculate the two-way travel times, i.e., the
time it takes the acoustic wave to reach a target and scatter back to a receiver. Here,
we want to use the stop-and-hop approximation to simplify the calculation.

```toml
[travel_time]
plugin = "stop_and_hop"
```

Various distortions can be applied to the acoustic waves. This is also an array of
plugins. Lets start by adding some energy loss due to geometric spreading (here we set
the power parameter for spherical spreading) and also due to acoustic attenuation.

```toml
[[distortion]]
plugin = "geometric_spreading"
power = 1

[[distortion]]
plugin = "anslie_mccolm_attenuation"
frequency = "centre"
```

Finally, lets configure the beampattern of the receivers. We could have included this
with each receiver definition above, but since we want the same for all receivers it is
easier to define this as a general plugin.

```toml
[[distortion]]
plugin = "rectangular_beampattern"
width = 0.015
height = 0.03
transmit = false
receive = true
frequency = "centre"
```

## Running the simulation

!!! Note
    The full configuration file described above can be found in the
    `examples/cli/simple_points` directory of the source code ([or viewed on
    GitHub](https://github.com/openSTB/simulator/tree/main/examples/cli/simple_points)).

This configuration file can now be passed to the simulator CLI to perform the
simulation. This is a simple CLI to use:

```console
openstb-sim run simple_points.toml
```

While the simulation is in progress, you can view the Dask diagnostic dashboard to view
the progress (assuming you added the `dask-diagnostics` set of optional dependencies
when installing) by going to [http://127.0.0.1:8787](http://127.0.0.1:8787).


## Viewing the results

When the simulation is complete, a NumPy file (or MATLAB file if you picked that option)
with the results will be in the same directory as the configuration file. For the NumPy
option, a simple Python script can be found in the `examples/cli/simple_points`
directory of the source code to plot the results.

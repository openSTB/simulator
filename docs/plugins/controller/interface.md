---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Controller interface

The base class defining the interface expected of a controller is [plugin.abc.Controller][openstb.simulator.plugin.abc.Controller].
For type hinting purposes, this is a [generic interface][generics] which allows the type
of the configuration class to be specified via subscription.


## Configuration class

Each controller will require a specific set of plugins to configure it. The plugin must
provide a property `config_class` which returns the configuration class to use when
running the controller. This class must be a mapping from string keys to values, with a
standard [TypedDict][typing.TypedDict] recommended to provide detailed type information
about each entry. For example, the included [`SimplePointSimulation`][openstb.simulator.controller.simple_points.SimplePointSimulation]
controller specifies the [`SimplePointConfig`][openstb.simulator.controller.simple_points.SimplePointConfig]
class for its configuration.

!!! Note
    The `config_class` property must return the class, not an instance of the class.


## Run method

A controller must provide a `run` method which is given an instance of its configuration
class. This is responsible for performing the corresponding simulation. No return value
is expected.


## Example

When implementing a controller, we first need to define the configuration structure it
requires. It is recommended to use a [`TypedDict`][typing.TypedDict] to enable type
hinting when writing the code that implements the controller itself.

As this example is minimalist, we only provide three configuration options: the cluster
to run on, the signal in use, and a list of the point targets to simulate the response
from.

```python
from typing import TypedDict

from openstb.simulator.plugin import abc


class MyControllerConfig(TypedDict):
    dask_cluster: abc.DaskCluster
    """The cluster to run the simulation on."""

    signal: abc.Signal
    """The signal transmitted by the sonar."""

    targets: list[abc.PointTargets]
    """The point targets to simulate."""
```

Next we write a function to calculate the response from an array of point targets. In
some cases this could be written as a method of the plugin class, but bear in mind that
Dask has to [serialise code and data][serialization] to transfer them to the worker
nodes. The instance of the plugin class (or one of its properties) may not be
serialisable, or it may store large amounts of data which both slow down the
serialisation process and consume worker memory.

In this example, the function takes the positions of the point targets and the
Fourier-domain representation of the transmitted signal. It calculates the two-way
travel time from the origin to the targets using a fixed sound speed of 1500m/s, applies
the corresponding phase shifts to create delayed copies of the signal, and sums these
responses over all the targets.

```python
import numpy as np


def simulate_chunk(
    position: np.ndarray, f: np.ndarray, signal_f: np.ndarray
) -> np.ndarray:
    # Calculate range and from that two-way travel time.
    r = np.sqrt(np.sum(position**2, axis=-1))
    t = 2 * r / 1500.0

    # Apply a phase shift corresponding to each travel time.
    # We need to expand the signal dimensionality for broadcasting.
    # E will have a shape (N_samples, N_targets).
    E = signal_f[:, np.newaxis] * np.exp(-2j * np.pi * f[:, np.newaxis] * t)

    # Sum over all targets in this chunk and remove the now-redundant dimension.
    return np.sum(E, axis=-1).squeeze()
```

We can now write our controller. The initialiser takes one parameter, the size of the
chunks to divide the point targets into when distributing the simulation tasks across
the cluster. The `config_class` property returns the configuration class we defined
earlier. The `run` method loads all point targets in chunks of the desired size, submits
tasks to the cluster to run our simulation function on these chunks, submits another
task to sum the results of all the chunks, and waits until the result is complete.

Note that when defining the class we specify the type of the configuration structure as
a generic when we inherit the base class.

```python
import logging


class MyController(abc.Controller[MyControllerConfig]):
    def __init__(self, points_per_chunk: int):
        self.points_per_chunk = points_per_chunk

    @property
    def config_class(self):
        return MyControllerConfig

    def run(self, config: MyControllerConfig):
        logger = logging.getLogger(__name__)

        # Load the target positions in chunks.
        positions = []
        logger.info("Loading target positions")
        for target in config["targets"]:
            N_points = len(target)
            for n in range(0, N_points, self.points_per_chunk):
                if (n + self.points_per_chunk) < N_points:
                    count = self.points_per_chunk
                else:
                    count = -1

                positions.append(target.get_chunk(n, count)[0])

        # Create a client to submit jobs to the cluster with.
        logger.info("Initialising Dask cluster")
        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

        # Sample the signal at 30kHz with a baseband carrier frequency of 100kHz.
        t = np.arange(0, 100e-3, 1/30e3)
        signal = config["signal"].sample(t, 100e3)

        # Convert into the Fourier domain.
        f = np.fft.fftfreq(len(t), 1/30e3) + 100e3
        signal_f = np.fft.fft(signal)

        # Schedule our simulation function to be run on each chunk. This returns
        # a list of Futures which can be used to find the status of each task
        # and retrieve the result when complete.
        logger.info("Submitting simulation tasks")
        results = client.map(simulate_chunk, positions, f=f, signal_f=signal_f)

        # Schedule the results to be summed. Note we do not retrieve the results
        # ourselves, but let Dask automatically transfer them to the worker
        # which will do the final sum.
        summed = client.submit(np.sum, results, axis=0)

        # We can now remove our reference to the futures for the initial results.
        # We no longer need it ourselves, and this lets Dask manage its lifetime.
        # When no other tasks depend on a Future, it will be removed from the
        # cluster.
        del results

        # Get the final result. This will wait until the task has completed.
        trace_f = summed.result()
        del summed

        # We can now return the result to the time domain.
        trace = np.fft.ifft(trace_f)
        logger.info("Simulation complete")
```

If the above code is saved into a local file then we can configure the simulation to
load the controller from that file. If we saved it to `example_controller.py`, then the
following TOML configuration will use the controller to run a simulation with two blocks
of point targets, one to starboard and the other to port:

```toml
[controller]
plugin = "MyController:example_controller.py"
points_per_chunk = 10

[dask_cluster]
plugin = "local"
workers = 8
total_memory = 0.4
dashboard_address = ":8787"

[signal]
plugin = "lfm_chirp"
f_start = 90e3
f_stop = 110e3
duration = 0.015
rms_spl = 190

[[targets]]
plugin = "random_point_rectangle"
seed = 106714187151181
Dx = 5
Dy = 120
centre = [0, 75, 10]
normal = [0, 0, -1]
point_density = 4
reflectivity = 0.06

[[targets]]
plugin = "random_point_rectangle"
seed = 8967190659810
Dx = 5
Dy = 120
centre = [0, -75, 10]
normal = [0, 0, -1]
point_density = 4
reflectivity = 0.06
```

The CLI can then be used to run this simulation with `openstb-sim run
example_controller.toml`. As this is a very simple example, it should not take long to
run.

Loading the controller directly from a file can be useful during development. For
wider use, it is more convenient to install the controller into the environment as part
of a standard Python package and register the plugin using an entry point. After
creating a standard [pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
file to package your project, add the following section to register an entry point which
maps a plugin name to the class:

```toml
[project.entry-points."openstb.simulator.controller"]
my_controller = "my_package.controller:MyController"
```

The simulation configuration can then be changed to use this plugin name:

```toml
[controller]
plugin = "my_controller"
points_per_chunk = 10
```

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Controller plugin helpers

Many simulation controllers will operate in a similar manner. The
[openstb.simulator.controller.base][] module contains base classes which can be
inherited by specific plugins to implement common functionality.


## Looping controller

A common simulation structure is to loop over a set of variables, for example ping and
receiver. In each loop the responses from a chunk of targets are simulated with given
parameters, and these responses aggregated with a reduction tree to form the final
output for that loop.

The [LoopController][openstb.simulator.controller.base.LoopController] base class was
created to assist with this structure of simulation. A plugin can inherit from this
class, set some required attributes and call its
[loop_run][openstb.simulator.controller.base.LoopController.loop_run] method to do the
simulation. This takes care of submitting the simulation tasks, doing so by monitoring
the number of tasks currently scheduled on the cluster and trying to keep this within a
specified pair of thresholds. It also monitors the execution of the tasks, reporting the
error and cancelling the remaining tasks if one fails.

The following attributes must be set by the inheriting class:

* [loop_chunks][openstb.simulator.controller.base.LoopController.loop_chunks]
* [loop_params][openstb.simulator.controller.base.LoopController.loop_params]
* [loop_simulate][openstb.simulator.controller.base.LoopController.loop_simulate]

The following attributes are optional:

* [loop_lower_threshold][openstb.simulator.controller.base.LoopController.loop_lower_threshold]
* [loop_upper_threshold][openstb.simulator.controller.base.LoopController.loop_upper_threshold]
* [loop_log_message][openstb.simulator.controller.base.LoopController.loop_log_message]


### Example

As an example, we will simulate the response when applying different vertical position
and timing errors to a receiver. The core simulation is below; note that the `targets`
parameter is designed to take one item from the iterator returned by
[target_chunk_iterator][openstb.simulator.target.points.target_chunk_iterator].

```python
import numpy as np

def simulate_chunk(
    targets: tuple[int, int, np.ndarray, np.ndarray],
    params: tuple[np.ndarray, np.ndarray, float, float]
) -> np.ndarray:
    """Perform one chunk of the simulation.

    Parameters
    ----------
    targets
        A tuple of (index, index, position, reflectivity) where only the
        position is used.
    params
        A (frequency, signal, vertical_error, timing_error) tuple.

    Returns
    -------
    np.ndarray
        The result of this simulation chunk.

    """
    _, _, position, _ = targets
    f, signal, vert, timing = params

    # Calculate range and from that two-way travel time, taking into account
    # the vertical error and timing error.
    r = np.sqrt(np.sum((position - [0, 0, vert])**2, axis=-1))
    t = 2 * r / 1500.0
    t -= timing

    # Apply a phase shift corresponding to each travel time.
    # We need to expand the signal dimensionality for broadcasting.
    # E will have a shape (N_samples, N_targets).
    E = signal[:, np.newaxis] * np.exp(-2j * np.pi * f[:, np.newaxis] * t)

    # Sum over all targets in this chunk and remove the redundant dimension.
    return np.sum(E, axis=-1).squeeze()
```

We will also define a function that in a real simulation would store the results to
disk.

```python
def store_result(result: np.ndarray, key: tuple[float, float]):
    """Store the simulation result.

    Parameters
    ----------
    result
        The result of some simulation chunks.
    key
        The (vertical_error, timing_error) key to save under.

    """
    ...
```

The configuration class for this example is very simple.

```python
from typing import TypedDict

from openstb.simulator.plugin import abc

class MyLoopConfig(TypedDict):
    dask_cluster: abc.DaskCluster
    """The cluster to run the simulation on."""

    signal: abc.Signal
    """The signal transmitted by the sonar."""

    targets: list[abc.PointTargets]
    """The point targets to simulate."""

```

The plugin implementation then inherits from `LoopController`. Note that this has
multiple generic types: the simulation chunk type, the simulation parameter type, the
simulation result type and the configuration type.

When calling `loop_run`, we set the variables in the order vertical then timing. The
loops will be performed in this order, and the loop key passed to our implementation
functions will also be a tuple (vertical, timing) with the values for the current loop.

```python
import distributed

from openstb.simulator.controller.base import LoopController
from openstb.simulator.target.points import target_chunk_iterator
from openstb.simulator.utils.reduction import DaskReductionTree

class MyLoopController(
    LoopController[
        tuple[int, int, np.ndarray, np.ndarray], # Chunk given to simulate function
        tuple[np.ndarray, np.ndarray, float, float], # Parameters for simulate function
        np.ndarray, # Output of simulate function
        MyLoopConfig # Configuration class
    ]
):

    @property
    def config_class(self):
        return MyLoopConfig

    def run(self, config: MyLoopConfig):
        # Create a client to submit jobs to the cluster with.
        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

        # Sample the signal at 30kHz with a baseband frequency of 100kHz.
        t = np.arange(0, 100e-3, 1/30e3)
        signal = config["signal"].sample(t, 100e3)

        # Convert into the Fourier domain.
        f = np.fft.fftfreq(len(t), 1/30e3) + 100e3
        signal_f = np.fft.fft(signal)

        # Specify the function to do the simulation.
        self.loop_simulate = simulate_chunk

        # Create a function which takes the loop key and returns an
        # iterator over the chunks of targets to simulate.
        def loop_chunks(loop_key: tuple[int, int]):
            return target_chunk_iterator(config["targets"], 20)

        self.loop_chunks = loop_chunks

        # Create a function which takes the loop key and returns the
        # parameters to use.
        def loop_params(loop_key: tuple[int, int]):
            return (f, signal_f, loop_key[0], loop_key[1])

        self.loop_params = loop_params

        # Define a function to take the output from the reduction tree and
        # schedule the result to be stored. Note that we add the future
        # corresponding to the store to the loop_futures set so that the
        # simulation loop will track it.
        def store(future: distributed.Future, loop_key: tuple[int, int]):
            self.loop_futures.add(
                client.submit(store_result, future, loop_key)
            )

        # Create a reduction tree to sum the results from each chunk.
        rtree = DaskReductionTree(client, store, np.sum, reduce_kwargs={"axis": 0})

        # And run the loop. In this case, we tell the helper class to
        # simulate 5 vertical errors; it will use np.arange(5) internally.
        # We explicitly provide the timing errors.
        self.loop_run(client, loop_vars=(5, np.arange(-2, 3) * 1e-6), rtree=rtree)

        # The simulation is complete so we can terminate the cluster.
        config["dask_cluster"].terminate()
```

This plugin could then be used with a configuration like the following.

```toml
[controller]
plugin = "MyLoopController:example_controller.py"

[dask_cluster]
plugin = "local"
workers = 8
total_memory = 0.4
dashboard_address = ":8778"

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
Dy = 5
centre = [0, 75, 10]
normal = [0, 0, -1]
point_density = 1
reflectivity = 0.06
```

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Simple point targets

**Plugin name**: `simple_points`
<br>
**Implementation**: [`openstb.simulator.controller.simple_points.SimplePointSimulation`][]

The `simple_points` controller simulates the scene as a collection of infinitesimally
small point targets. These have no aspect dependence and cause no shadows. The
simulation is performed in the frequency domain. For a two-way travel time $\tau$ from a
transmitter to a target and back to a receiver, a phase shift of the signal spectrum
$S_{rx}(f) = S(f)\exp(-2j\pi f \tau)$ can be used to model the delayed signal.


## Result filename

The results of the simulation are stored in a [Zarr](https://zarr.readthedocs.io/) file.
The filename to use must be specified by the `result_filename` parameter to the plugin.
When initialised, the plugin checks if this filename already exists. If so, an error is
raised to prevent overwriting earlier results. A second check is made just before the
simulation starts in case the file has been created in the meantime (for example, while
the other plugins were being initialised). An error is also raised in this case.

Note that a result converter plugin can be configured to take the Zarr file and convert
it to your preferred format. The controller does not perform any checks whether this
final output already exists; it is up to the result converter plugin to check for this
and handle it accordingly.


## Chunk size

The point targets are broken into chunks for simulation. The maximum number of points in
each chunk must be set by the `points_per_chunk` parameter. This should be small enough
to fit at least two chunks in the memory of each worker. However, note that the smaller
the chunks are the more chunks are required for the simulation, and thus the more
overhead there is on the cluster.


## Sampling properties

The simulation produces samples of the received pressure in baseband. The sampling rate
of the result in Hertz must be specified via the `sample_rate` parameter. The carrier
frequency used to convert the samples to baseband must be specified (also in Hertz) by
the `baseband_frequency` parameter.



## Task submission thresholds

The controller attempts to submit simulation tasks to the cluster in such a way as to
keep the number of in-progress and pending tasks within a set range. If one task
finishes and there is no subsequent task waiting, then the workers will have to idle
until a task becomes available. This reduces the efficiency of the simulation. Tracking
and managing the tasks requires overhead on the cluster scheduler, and so having too
many tasks can also be suboptimal.

The thresholds for the number of tasks the cluster should have are specified as factors
of the number of workers in the cluster. The minimum number of tasks is set by the
`task_lower_threshold` parameter which defaults to 2, and the maximum number of tasks is
set by the `task_upper_threshold` parameter which defaults to 3. When the number of
tasks on the cluster drops below the lower threshold, the controller adds more tasks
until the upper threshold is reached. Note that the thresholds are not hard limits and
that due to timing the number of tasks may be outside the set range.


## Reduction tree size

The results of each chunk of the simulation need to be added together and eventually
written to disk. It is not desirable to add each chunk to the result on disk as it
becomes available as the read-add-write procedure would cause a bottleneck. Instead the
controller employs a [reduction tree](https://en.wikipedia.org/wiki/Reduction_operator)
to iteratively sum the results from different chunks. This accumulates the result,
allowing the original results to be freed from memory. For a node count of two, the
first level of the reduction might be

\[
\begin{aligned}
a_{1,1} &= s_1 + s_2, \\
a_{1,2} &= s_3 + s_4.
\end{aligned}
\]

A second level of reduction can then be applied to get

\[
\begin{aligned}
a_{2,1} &= a_{1,1} + a_{1,2}, \\
&= s_1 + s_2 + s_3 + s_4.
\end{aligned}
\]

After a certain number of levels, the accumulated result is added to the result on disk.
The number of nodes accumulated at each level is set by the `reduction_node_count`
parameter (which defaults to 4) and the number of levels to reduce before writing to
disk is set by the `reduction_levels` parameter (which defaults to 3).

!!! Note
    Floating-point addition is [not necessarily associative](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems),
    that is, $(s_1+s_2)+s_3$ may give a different result to $s_1 + (s_2 + s_3)$ due to
    precision limitations. This means that running the same simulation twice may yield
    results with slightly differing values depending on the order results are summed.

    The order the reduction tree is summed is deterministic, so the only differences can
    come from the order the accumulated results are added to the value on disk. With
    double-precision floats being used, any such non-associative behaviour is not
    expected to have any noticable impact on the usability of the simulation results.

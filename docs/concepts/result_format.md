---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Result format

Internally, a simulation controller stores the results in a fixed format suitable for
its operation. To convert the results to the desired format, a [result converter
plugin](../plugins/result_converter/overview.md) may be configured. Typically, if the
converter succeeds then the original results in the internal format are deleted by the
controller.


## Enumeration

The [openstb.simulator.plugin.abc.ResultFormat][] class is an [enumeration][enum]
providing constants to identify the type of result format used by a simulation
controller. One of the values from this enumeration is provided to the result converter
plugin to allow it to determine whether they are able to convert the results to the
desired format.


## Zarr baseband pressure

The only result format currently defined is the *Zarr baseband pressure* format
(corresponding to the [ZARR_BASEBAND_PRESSURE][openstb.simulator.plugin.abc.ResultFormat.ZARR_BASEBAND_PRESSURE]
enumeration value). This uses a [Zarr][] group to store the results. Three variables are
stored within the group:

* `pressure`: a three-dimensional array of shape (pings, receivers, samples) containing
  the pressure recorded by the receivers on each ping. The values are Pascals in the
  complex baseband.

* `sample_time`: a one-dimensional array of length samples containing the sample time of
  each pressure measurement. These are given in seconds relative to when the ping
  transmission began.

* `ping_start_time`: a one-dimensional array of length pings containing the start time
  of each ping transmission. These are given in seconds relative to the beginning of the
  trajectory.

In addition, the following attributes are set on the group:

* `baseband_frequency`: the frequency (in Hertz) that was used to convert the recorded
  pressures into the complex baseband.

* `sample_rate`: the sample rate (in Hertz) that the pressure was recorded at.

[Zarr]: https://zarr.readthedocs.io/

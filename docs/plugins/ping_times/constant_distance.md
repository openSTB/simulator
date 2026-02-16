---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Constant distance

**Plugin name**: `constant_distance`
<br>
**Implementation**: [`openstb.simulator.system.ping_times.ConstantDistance`][]

This plugin models a system operating with a constant ping repetition distance, i.e.,
which adapts the interval between pings to compensate for changes in velocity. The
desired distance between pings is given by the `distance` parameter, a floating-point
value in metres. The `start_offset` parameter gives the distance (as a floating-point
value in metres) between the start of the trajectory and the first ping being
transmitted. The minimum distance between transmission of the final ping and the end of
the trajectory is given by the `end_offset` parameter, another floating-point value in
metres. These three parameters are required; the two offsets can be set to zero if not
relevant.

Internally, the plugin calculates the mean speed of the system throughout the
trajectory. Dividing the desired ping distance by this mean speed gives the mean time
interval between pings. It then samples the trajectory a set number of times in each of
these intervals, giving a mapping between time and cumulative distance travelled along
the trajectory. This mapping is interpolated with a [piecewise cubic hermite
interpolating polynomial (PCHIP)][PCHIP] interpolator to find the time corresponding to
each desired distance.

The `sampling_factor` parameter sets the number of samples taken within each mean time
interval to generate the mapping. It is an integer value which defaults to 10 if not
explicitly set.

[PCHIP]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

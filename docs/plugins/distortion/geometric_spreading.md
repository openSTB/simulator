---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Geometric spreading

**Plugin name**: `geometric_spreading`
<br>
**Implementation**: [`openstb.simulator.distortion.environmental.GeometricSpreading`][]

As a pressure wave spreads through the water, it covers a larger surface area. This
means that the energy density of the wave reduces with distance. The geometric spreading
as calculated by this plugin assumes the environment is lossless.


## Parameters

The plugin has one required parameter, `power`. This is the power that the range is
raised to when calculating the one-way amplitude scaling factor, i.e., the scaling
factor $a = 1/r^\text{power}$. As this is applied to the amplitude, a power of 1
corresponds to spherical spreading ($1/r$ in amplitude, $1/r^2$ in intensity) and a
power of 0.5 corresponds to cylindrical spreading ($1/\sqrt{r}$ in amplitude, $1/r$ in
intensity).

Two optional parameters may also be given:

* `transmit`: whether to apply spreading losses for the length of the transmit path.
* `receive`: whether to apply spreading losses for the length of the receive path.

These both default to true, i.e., the following configurations produce identical
results.

```toml
[distortion]
plugin = "geometric_spreading"
power = 1
```

```toml
[distortion]
plugin = "geometric_spreading"
power = 1
transmit = true
receive = true
```

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Rectangular beampattern

**Plugin name**: `rectangular_beampattern`
<br>
**Implementation**: [`openstb.simulator.distortion.beampattern.RectangularBeampattern`][]

This plugin applies an angle-dependent scaling corresponding to the beampattern from an
ideal rectangular beampattern. For a size $D$ in one axis, the scaling factor is given
by

\[
\alpha(\theta) = \operatorname{sinc}\left(\frac{D \sin\theta}{\lambda}\right).
\]


## Parameters

The plugin has several required parameters:

* `width`: the width of the aperture in metres
* `height`: the height of the aperture in metres
* `transmit`: whether to apply the beampattern for the transmission angle
* `receive`: whether to apply the beampattern for the reception angle
* `frequency`: which frequency or frequencies to calculate the scaling factor at:
    * `min`: calculate at the lowest frequency in the simulation.
    * `max`: calculate at the highest frequency in the simulation.
    * `centre`: calculate at the centre frequency (i.e., the midpoint of `min` and `max`).
    * `all`: calculate the attenuation separately for each frequency being simulated.

For example, to apply a beampattern for a transmitter than is 2cm wide and 3cm high
using the centre frequency:

```toml
[distortion]
plugin = "rectangular_beampattern"
width = 2e-2
height = 3e-2
transmit = true
receive = false
frequency = "centre"
```

The plugin also has two optional parameters, `horizontal` and `vertical`. These are both
boolean values which default to true, and indicate whether to include the horizontal and
vertical components of the beampattern. If we only wanted to apply the horizontal
beampattern from the previous example, we could modify it to

```toml
[distortion]
plugin = "rectangular_beampattern"
width = 2e-2
height = 3e-2
transmit = true
receive = false
frequency = "centre"
horizontal = true
vertical = false
```

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Doppler effect

**Plugin name**: `doppler`
<br>
**Implementation**: [`openstb.simulator.distortion.doppler.DopplerDistortion`][]

The movement of the platform while transmitting the signal and capturing the echoes
causes a Doppler-based distortion. The Doppler shift of each frequency in the signal
differs, resulting in a dilation or contraction of the spectrum. The size of the
distortion is proportional to the component of the platform velocity in the direction of
the target, i.e., a target that is perfectly broadside experiences no distortion.

This plugin applies a Doppler distortion using the following steps:

1. Calculate the speed of the transmitter and receiver in the direction of the target
   (positive being towards the target).

2. Calculate the Doppler scale factor $\eta = \dfrac{1 + v_\text{rx}/c}{1 - v_\text{tx}/c}$.

3. Distort the spectrum according to $S'(f) = \dfrac{S(f/\eta)}{\sqrt{\eta}}$.


## Parameters

The plugin has no required parameters:

```toml
[distortion]
plugin = "doppler"
```

The plugin only takes one optional parameter, the Boolean `calculate_c_rx` which
defaults to false. When false, the sound speed $c$ is only measured at the time
and position of the transmission. When true, the sound speed is measured separately at
the reception time and position of the echo from every target. Unless the sound speed
varies quickly, the default false should be sufficient.

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Constant scattering

**Plugin name**: `constant`
<br>
**Implementation**: [`openstb.simulator.scattering.constant.ConstantScattering`][]

The simplest scattering model is to scale the incident intensity by a constant factor
$\alpha$, i.e, the intensity of the scattered wave

\[
I_s = \alpha I_i
\]

for an incident intensity $I_i$. This is independent of incident and scattered angle and
frequency.


## Parameters

The only parameter the plugin takes is `scale_factor`, corresponding to $\alpha$ in
the previous section.

```toml
[scattering]
plugin = "constant"
scale_factor = 0.05
```

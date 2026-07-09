---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Lambertian scattering

**Plugin name**: `lambertian`
<br>
**Implementation**: [`openstb.simulator.scattering.lambertian.LambertianScattering`][]

[Lambertian scattering](https://en.wikipedia.org/wiki/Lambertian_reflectance) occurs
from an ideal diffusive (matte) surface. The intensity of the scattered wave $I_s$ is
the same at all scattering angles. This intensity is calculated as

\[
I_s = \alpha I_i \cos\theta
\]

where $I_i$ is the intensity of the incident wave, $\theta$ is the angle between the
incident wave and the normal of the surface, and $\alpha$ is an additional scaling
factor which can be used to model the scattering strength of the interface material.


## Parameters

The only parameter the plugin takes is `scale_factor`, corresponding to $\alpha$ in the
previous section. This defaults to 1, i.e., the following two configurations give an
identical scattering model.

```toml
[scattering]
plugin = "lambertian"
```

```toml
[scattering]
plugin = "lambertian"
scale_factor = 1
```

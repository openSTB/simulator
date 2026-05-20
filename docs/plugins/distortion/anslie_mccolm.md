---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Anslie-McColm attenuation

**Plugin name**: `anslie_mccolm_attenuation`
<br>
**Implementation**: [`openstb.simulator.distortion.environmental.AnslieMcColmAttenuation`][]

The interaction between the sound wave and the water causes an attenuation of the energy
in the wave. Anslie and McColm[^paper] model the impact of viscous drag and chemical
relaxation due to both boric acid and magnesium sulphate. Along with the frequency and
operating depth, the temperature, salinity and pH of the water are parameters for this
model. For common values, only the salinity has a significant affect. Broadly speaking,
the chemical relaxation of boric acid dominates below 1kHz, the chemical relaxation of
magnesium sulphate dominates from 10kHz to 100kHz and the viscous drag dominates above
200kHz.

[^paper]: Michael A. Ainslie and James G. McColm, "A simplified formula for viscous and
    chemical absorption in sea water". *J. Acoust. Soc. Am.*. 1 March 1998, volume 103
    number 3, pages 1671–1672. DOI: [10.1121/1.421258](https://doi.org/10.1121/1.421258)


## Model

The model in the paper takes the frequency in kHz and the depth in kilometres. It
characterises the chemical relaxations in terms of an attenuation coefficient and a
relaxation frequency. Boric acid has a relaxation frequency

\[
f_1 = 0.78\sqrt{(S/35)}\,e^{T/26}
\]

and a coefficient

\[
A = 0.106 e^{(\mathrm{pH} - 8)/0.56}
\]

for salinity S and temperature T. Similarly, the relaxation frequency of magnesium
sulphate is

\[
f_2 = 42e^{T/17}
\]

and its attenuation coefficient is

\[
B = 0.52 \left(1 + \frac{T}{43}\right) \left(\frac{S}{35}\right) e^{-D/6}
\]

for depth D. Finally, the viscous drag has an attenuation coefficient of

\[
C = 0.00049 e^{-(T/27 + D/17)}.
\]

The total attenuation, in dB/km, is then given by

\[
\alpha_a(f) = \frac{Af_1f^2}{f_1^2 + f^2} + \frac{Bf_2f^2}{f_2^2 + f^2} + Cf^2.
\]


## Parameters

The only required parameter is `frequency` which must be one of the following strings:

* `min`: calculate the attenuation at the lowest frequency in the simulation.
* `max`: calculate the attenuation at the highest frequency in the simulation.
* `centre`: calculate the attenuation at the centre frequency (i.e., the midpoint of
  `min` and `max`).
* `all`: calculate the attenuation separately for each frequency being simulated.

```toml
[distortion]
plugin = "anslie_mccolm_attenuation"
frequency = "centre"
```

Three optional parameters may also be given:

* `pH`: the pH level of the water.
* `transmit`: whether to apply attenuation corresponding to the length of the transmit
  path.
* `receive`: whether to apply attenuation corresponding to the length of the receive
  path.

The default settings correspond to the following configuration:

```toml
[distortion]
plugin = "anslie_mccolm_attenuation"
frequency = "centre"
pH = 8.0
transmit = true
receive = true
```

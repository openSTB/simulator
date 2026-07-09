---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Scattering model interface

The base class defining the interface expected of a distortion plugin is
[plugin.abc.ScatteringModel][openstb.simulator.plugin.abc.ScatteringModel]. Plugins are
registered under the `openstb.simulator.scattering_model` entry point.


## Apply method

The plugin must provide an [`apply`][openstb.simulator.plugin.abc.ScatteringModel.apply]
method which applies the distortion to the current signal. This is given a number of
parameters:

* The frequencies being simulated.

* The Fourier coefficients corresponding to the current signal in the complex baseband.

* The time the scattering occurred, in seconds relative to the start of the trajectory.

* The position where the scattering occurred and the normal of the interface at that
  position.

* The incident and scattering vectors. These define the direction the acoustic wave was
  travelling when it hit the interface and the direction the scattered wave which
  reaches the receiver was travelling when it left the interface, respectively.

* The [Environment](../environment/overview.md) plugin defining the parameters of the
  environment the system is operating in.

* The baseband frequency used to convert the transmitted signal into baseband.

* The bounds of the transmitted signal (which are typically narrower than the bounds of
  the frequencies being simulated).

This method must calculate the scattering strength for the given scenario and return a
copy of the incident signal Fourier coefficients modified accordingly.


## Example

The following plugin would calculate the amplitude of the scattered wave by multiplying
the amplitude of the incident wave by a scaling factor (as opposed to the included
`constant` plugin, which scales the intensity).


```python
import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin.abc import Environment, ScatteringModel


class ConstantAmplitudeScattering(ScatteringModel):
    """Interface with a constant amplitude scattering strength.

    This models the amplitude of the scattered wave as the amplitude of the incident
    wave multiplied by a constant scale factor.

    """

    def __init__(self, scale_factor: float):
        """
        Parameters
        ----------
        scale_factor
            Multiplicative scale factor to get the scattered amplitude from the incident
            amplitude.

        """
        self.scale_factor = scale_factor

    def apply(
        self,
        f: ArrayLike,
        S: ArrayLike,
        time: ArrayLike,
        position: ArrayLike,
        normal: ArrayLike,
        incident_vector: ArrayLike,
        scattering_vector: ArrayLike,
        environment: Environment,
        baseband_frequency: float,
        signal_frequency_bounds: tuple[float, float],
    ) -> np.ndarray:
        return np.asarray(S) * self.scale_factor
```

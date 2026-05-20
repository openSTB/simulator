---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Distortion plugin interface

The base class defining the interface expected of a distortion plugin is
[plugin.abc.Distortion][openstb.simulator.plugin.abc.Distortion]. Plugins are registered
under the `openstb.simulator.distortion` entry point.


## Apply method

The plugin must provide an [`apply`][openstb.simulator.plugin.abc.Distortion.apply]
method which applies the distortion to the current signal. This is given a number of
parameters:

* The time the ping was transmitted, in seconds relative to the start of the trajectory.

* The frequencies being simulated.

* The Fourier coefficients corresponding to the current signal in the complex baseband.

* The baseband frequency used to convert the transmitted signal into baseband.

* Parameters of the environment the system is operating in.

* The bounds of the transmitted signal (which are typically narrower than the bounds of
  the frequencies being simulated).

* The travel time result for each target being simulated.

It must then apply the distortion to the given Fourier coefficients and return the
updated version.


## Example

The following example plugin calculates and applies a frequency-dependent scale factor.

```python
import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult


class FrequencyDependentScaling(Distortion):
    """Frequency-dependent amplitude scaling."""

    def __init__(self, factor: float):
        """
        Parameters
        ----------
        factor
            The factor to increase the amplitude by over the simulated
            frequencies. The lowest simulated frequency will be unchanged,
            and the others will be linearly scaled such that the highest
            simulated frequency is increased by 1 + factor.

        """
        self.factor = factor

    def apply(
        self,
        ping_time: float,
        f: ArrayLike,
        S: ArrayLike,
        baseband_frequency: float,
        environment: Environment,
        signal_frequency_bounds: tuple[float, float],
        tt_result: TravelTimeResult,
    ) -> np.ndarray:
        # Range of frequencies being simulated.
        f_range = np.max(f) - np.min(f)

        # Relative frequencies within this range.
        df = (f - np.min(f)) / f_range

        # Calculate the scale factor.
        scale = 1 + (df * self.factor)

        # Frequency is the middle axis, so add dummy axes and apply.
        return S * scale[None, :, None]
```

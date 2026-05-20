---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Signal plugin interface

The base class defining the interface expected of a result converter plugin is
[plugin.abc.Signal][openstb.simulator.plugin.abc.Signal]. Plugins are registered under
the `openstb.simulator.signal` entry point.


## Properties

The plugin must have the following properties set:

* [`duration`][openstb.simulator.plugin.abc.Signal.duration]: the duration of the signal
  in seconds.

* [`maximum_frequency`][openstb.simulator.plugin.abc.Signal.maximum_frequency]: the
  highest instantaneous frequency within the signal.

* [`minimum_frequency`][openstb.simulator.plugin.abc.Signal.minimum_frequency]: the
  lowest instantaneous frequency within the signal.

All these properties must be floating-point values. Note that the base class defines
them as abstract methods with the [@property][property] decorator. If your plugin
inherits from the base class, you must implement these in the same way. The example
plugin below demonstrates this.


## Baseband samples

The plugin must provide a [`sample`][openstb.simulator.plugin.abc.Signal.sample] method
which returns samples of the signal in the baseband. This will be given an array of
times to sample at; a time of zero corresponds to the start of transmission. It will
also be given the frequency to use when basebanding the samples. It must return an array
of samples of the same shape as the time input.


## Example

The following plugin would provide a signal that is the sum of two complex sinusoids.

```python
import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin import abc


class TwoSinusoids(abc.Signal):
    """Sum of two complex sinusoids."""

    def __init__(self, f1: float, f2: float, duration: float, amplitude: float):
        """
        Parameters
        ----------
        f1
            Frequency of one sinusoid.
        f2
            Frequency of the other sinusoid.
        duration
            Length of the signal.
        amplitude
            Amplitude (in Pascal) of each sinusoid.

        """
        self.f1 = f1
        self.f2 = f2
        self._duration = duration
        self.amplitude = amplitude

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def maximum_frequency(self) -> float:
        return max(self.f1, self.f2)

    @property
    def minimum_frequency(self) -> float:
        return min(self.f1, self.f2)

    def sample(self, t: ArrayLike, baseband_frequency: float) -> np.ndarray:
        # Ensure we have an array.
        t = np.atleast_1d(t)

        # Sum and scale the sinusoids.
        s = np.exp(2j * np.pi * (self.f1 - baseband_frequency) * t)
        s += np.exp(2j * np.pi * (self.f2 - baseband_frequency) * t)
        s *= self.amplitude

        # Clear values outside the duration.
        s[t < 0] = 0
        s[t > self._duration] = 0
        return s
```

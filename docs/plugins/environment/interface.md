---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Environment plugin interface

The base class defining the interface expected of an environmental plugin is
[plugin.abc.Environment][openstb.simulator.plugin.abc.Environment]. Plugins are
registered under the  `openstb.simulator.environment` entry point.


## Salinity

An environment plugin must provide a [`salinity`][openstb.simulator.plugin.abc.Environment.salinity]
method. This takes an array of times and positions and must return the salinity of the
water (in parts per thousand) at those points.


## Sound speed

An environment plugin must provide a [`sound_speed`][openstb.simulator.plugin.abc.Environment.sound_speed]
method. This takes an array of times and positions and must return the speed of sound in
the water (in metres per second) at those points.


## Temperature

An environment plugin must provide a [`temperature`][openstb.simulator.plugin.abc.Environment.temperature]
method. This takes an array of times and positions and must return the temperature of
the water (in degrees Celsius) at those points.


## Example

The following plugin models an environment where the speed of sound varies with a
constant gradient with respect to depth. The salinity and temperature are constant.

```python
import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin.abc import Environment

class LinearSpeed(Environment):
    """Environment with speed of sound varying linearly with depth."""

    def __init__(self,
        surface_speed: float,
        speed_gradient: float,
        temperature: float,
        salinity: float
    ):
        """
        Parameters
        ----------
        surface_speed
            The speed of sound at the surface of the water in metres per second.
        speed_gradient
            The gradient of the speed of sound, in metres per second per metre.
        temperature
            The temperature of the water in degrees Celsius.
        salinity
            The salinity of the water in parts per thousand.

        """
        self._surface_speed = surface_speed
        self._speed_gradient = speed_gradient
        self._temperature = temperature
        self._salinity = salinity

    def salinity(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        position = np.asarray(position)

        # Find the broadcast shape of the inputs, ignoring the final axis of position.
        bc_shape = np.broadcast_shapes(t.shape, position.shape[:-1])

        # Take our constant and turn it into an array of size 1 along all dimensions.
        return np.array(self._salinity).reshape([1] * len(bc_shape))

    def sound_speed(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        pos = np.asarray(position)
        speed = self._surface_speed + pos[..., 2] * self._speed_gradient
        return speed * np.ones_like(t)

    def temperature(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        position = np.asarray(position)
        bc_shape = np.broadcast_shapes(t.shape, position.shape[:-1])
        return np.array(self._temperature).reshape([1] * len(bc_shape))
```

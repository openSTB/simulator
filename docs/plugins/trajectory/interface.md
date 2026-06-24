---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Trajectory plugin interface

The base class defining the interface expected of a trajectory plugin is
[plugin.abc.Trajectory][openstb.simulator.plugin.abc.Trajectory]. Plugins are registered
under the `openstb.simulator.trajectory` entry point.


## Properties

The plugin must have the following properties set:

* [`duration`][openstb.simulator.plugin.abc.Trajectory.duration]: the duration of the
  trajectory in seconds.

* [`length`][openstb.simulator.plugin.abc.Trajectory.length]: the length of the
  trajectory in metres.

* [`start_time`][openstb.simulator.plugin.abc.Trajectory.start_time]: the start time as
  a Python [datetime.datetime][] instance in UTC.

All these properties must be floating-point values. Note that the base class defines
them as abstract methods with the [@property][property] decorator. If your plugin
inherits from the base class, you must implement these in the same way. The example
plugin below demonstrates this.


## Position

The plugin must provide a [`position`][openstb.simulator.plugin.abc.Trajectory.position]
method which returns the position of the system at particular times. The input will be
an array or array-like object of times in seconds since the start of the trajectory. The
output should be an equivalent array with an extra final dimension of size 3 containing
the x, y and z positions of the system at those times. Any requested values before the
trajectory starts or after it ends must be set to [nan][numpy.nan].


## Orientation

The plugin must provide an [`orientation`][openstb.simulator.plugin.abc.Trajectory.orientation]
method which returns the orientation of the system at particular times. The input will be
an array or array-like object of times in seconds since the start of the trajectory. The
output should be an equivalent array with an extra final dimension of size 4 containing
the components of a quaternion describing the rotation from the global axes to the
system axes at those times. The first component must be the scalar part and the
remaining three components the vector parts corresponding to the x, y and z axes in that
order. Any requested values before the trajectory starts or after it ends must be set to
[nan][numpy.nan].


## Velocity

The plugin must provide a [`velocity`][openstb.simulator.plugin.abc.Trajectory.velocity]
method which returns the velocity of the system at particular times. The input will be
an array or array-like object of times in seconds since the start of the trajectory. The
output should be an equivalent array with an extra final dimension of size 3 containing
the x, y and z components of the system velocity at those times. Any requested values
before the trajectory starts or after it ends must be set to [nan][numpy.nan].


## Example

The following plugin defines a trajectory with two linear segments.

```python
from datetime import datetime, timezone

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.simulator.plugin.abc import Trajectory
from openstb.simulator.util import quaternion_from_vectors


class TwoSegmentLinear(Trajectory):
    """A trajectory with two linear segments."""

    def __init__(self,
        start_position: ArrayLike,
        join_position: ArrayLike,
        end_position: ArrayLike,
        speed: float
    ):
        """
        Parameters
        ----------
        start_position
            Position the system starts at.
        join_position
            Position of the join between the segment.
        end_position
            Position the system ends at.
        speed
            The speed of the system in metres per second.

        """
        self._start_time = datetime.now(timezone.utc)

        # Store the parameters.
        self.start_position = np.asarray(start_position)
        self.join_position = np.asarray(join_position)
        self.end_position = np.asarray(end_position)
        self.speed = speed

        # Calculate some information about the first segment.
        self._diff1 = self.join_position - self.start_position
        self._length1 = float(np.linalg.norm(self._diff1))
        self._velocity1 = speed * self._diff1 / self._length1
        self._duration1 = float(self._length1 / speed)

        # And about the second.
        self._diff2 = self.end_position - self.join_position
        self._length2 = float(np.linalg.norm(self._diff2))
        self._velocity2 = speed * self._diff2 / self._length2
        self._duration2 = float(self._length2 / speed)

    @property
    def duration(self) -> float:
        return self._duration1 + self._duration2

    @property
    def length(self) -> float:
        return self._length1 + self._length2

    @property
    def start_time(self) -> datetime:
        return self._start_time

    def position(self, t: ArrayLike) -> np.ndarray:
        # Start with positions for segment 1 velocity.
        t = np.asarray(t)
        pos = self.start_position + t[..., np.newaxis] * self._velocity1

        # Replace second segment values.
        t2 = t - self._duration1
        pos2 = self.join_position + t2[..., np.newaxis] * self._velocity2
        second = t > self._duration1
        pos[second] = pos2[second]

        # And get rid of values outside the duration.
        invalid = (t < 0) | (t > self.duration)
        pos[invalid] = np.nan
        return pos

    def orientation(self, t: ArrayLike):
        # Start with orientation of segment 1.
        t = np.asarray(t)
        ori = np.full(t.shape + (4,), quaternion_from_vectors([1, 0, 0], self._diff1))

        # Replace second segment values.
        ori2 = np.full(t.shape + (4,), quaternion_from_vectors([1, 0, 0], self._diff2))
        second = t > self._duration1
        ori[second] = ori2[second]

        # And get rid of values outside the duration.
        invalid = (t < 0) | (t > self.duration)
        ori[invalid] = np.nan
        return quaternionic.array(ori)

    def velocity(self, t: ArrayLike) -> np.ndarray:
        t = np.asarray(t)

        # Initialise as invalid and set the valid components.
        vel = np.full(t.shape + (3,), np.nan)
        vel[(t >= 0) & (t < self._duration1)] = self._velocity1
        vel[(t >= self._duration1) & (t <= self.duration)] = self._velocity2

        return vel
```

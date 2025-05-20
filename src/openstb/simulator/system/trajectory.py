# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Trajectory plugins for the openSTB simulator."""

from datetime import datetime, timezone

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import Trajectory

_ = translations.load("openstb.simulator").gettext


class Linear(Trajectory):
    """An ideal linear trajectory."""

    def __init__(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        speed: float,
        start_time: datetime | str | int | None = None,
    ):
        """
        Parameters
        ----------
        start_position, end_position : array-like of 3 floats
            The start and end position of the trajectory in the global coordinate
            system.
        speed : float
            The speed of the system in metres per second.
        start_time : datetime.datetime, str, int, optional
            The time at which the trajectory starts. If a datetime instance is given, it
            will be converted to UTC. An string in the ISO 8601 format
            "YYYY-MM-DDTHH:MM:SS+ZZ:ZZ", where the "+ZZ:ZZ" represents the offset of the
            timezone, can be given. If the timezone offset is not given, it will be
            assumed to be UTC. An integer representing a UTC POSIX timestamp (seconds
            since midnight on 1 January 1970) can be given. If no start time is given,
            it is set to the time the trajectory instance is initialised.

        """
        self.start_position = np.asarray(start_position)
        if self.start_position.shape != (3,):
            raise ValueError(
                _("3 element vector required for linear trajectory start position")
            )
        self.end_position = np.asarray(end_position)
        if self.end_position.shape != (3,):
            raise ValueError(
                _("3 element vector required for linear trajectory end position")
            )

        # Calculate the length, velocity and duration of the trajectory.
        if not speed > 0:
            raise ValueError(_("speed of linear trajectory must be positive"))
        diff = self.end_position - self.start_position
        self._length = float(np.linalg.norm(diff))
        self._velocity = speed * diff / self._length
        self._duration = float(self._length / speed)

        # Calculate the orientation of the system. The cross product will be zero for
        # trajectories parallel to the x axis hence the special cases. We store the
        # result as a NumPy array here as quaternionic arrays cannot be pickled for
        # transfer between workers.
        hvec = diff / self._length
        dp = np.dot([1, 0, 0], hvec)
        if np.isclose(dp, 1):
            self._ori = np.array([1.0, 0.0, 0.0, 0.0])
        elif np.isclose(dp, -1):
            self._ori = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            angle = np.arccos(dp)
            axis = np.cross([1, 0, 0], hvec)
            axis /= np.linalg.norm(axis)
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            self._ori = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        # Calculate or convert the start time as needed.
        if start_time is None:
            self._start_time = datetime.now(timezone.utc)
        elif isinstance(start_time, str):
            raw = datetime.fromisoformat(start_time)
            if raw.tzinfo is None:
                self._start_time = raw.replace(tzinfo=timezone.utc)
            else:
                self._start_time = raw.astimezone(timezone.utc)
        elif isinstance(start_time, int):
            self._start_time = datetime.fromtimestamp(start_time, timezone.utc)
        else:
            self._start_time = start_time.astimezone(timezone.utc)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def length(self) -> float:
        return self._length

    @property
    def start_time(self) -> datetime:
        return self._start_time

    def position(self, t: ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        pos = self.start_position + t[..., np.newaxis] * self._velocity
        invalid = (t < 0) | (t > self._duration)
        pos[invalid] = np.nan
        return pos

    def orientation(self, t: ArrayLike) -> quaternionic.QArray:
        t = np.asarray(t)
        ori = np.full(t.shape + (4,), self._ori)
        invalid = (t < 0) | (t > self._duration)
        ori[invalid] = np.nan
        return quaternionic.array(ori)

    def velocity(self, t: ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        vel = np.full(t.shape + (3,), self._velocity)
        invalid = (t < 0) | (t > self._duration)
        vel[invalid] = np.nan
        return vel

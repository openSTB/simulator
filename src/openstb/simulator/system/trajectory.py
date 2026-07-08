# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Trajectory plugins for the openSTB simulator."""

from datetime import datetime, timezone

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import Trajectory
from openstb.simulator.utils.quaternion import quaternion_from_vectors

_ = translations.load("openstb.simulator").gettext


class Circular(Trajectory):
    """An ideal circular trajectory."""

    def __init__(
        self,
        centre: ArrayLike,
        radius: float,
        speed: float,
        clockwise: bool,
        num_circles: float = 1.0,
        start_angle: float = 0,
        start_time: datetime | str | int | None = None,
    ):
        """
        Parameters
        ----------
        centre
            The centre of the trajectory in the global coordinate system.
        radius
            The radius of the circle in metres.
        speed
            The speed of the system in metres per second.
        clockwise
            True if the system is travelling clockwise around the circle (from the x
            axis towards the y axis), false if it is travelling counter-clockwise (from
            the x axis towards the negative y axis).
        num_circles
            The number of complete circles within the trajectory.
        start_angle
            The angle (in degrees) from the centre of the circle to the system at the
            start of the trajectory. An angle of 0 corresponds to the vector between
            circle centre and system being parallel to the x axis, and an angle of 90
            to that vector being parallel to the y axis.
        start_time
            The time at which the trajectory starts. If a datetime instance is given, it
            will be converted to UTC. An string in the ISO 8601 format
            "YYYY-MM-DDTHH:MM:SS+ZZ:ZZ", where the "+ZZ:ZZ" represents the offset of the
            timezone, can be given. If the timezone offset is not given, it will be
            assumed to be UTC. An integer representing a UTC POSIX timestamp (seconds
            since midnight on 1 January 1970) can be given. If no start time is given,
            it is set to the time the trajectory instance is initialised.

        """
        self.centre = np.asarray(centre)
        if self.centre.shape != (3,):
            raise ValueError(
                _("3 element vector required for circular trajectory centre")
            )
        if not speed > 0:
            raise ValueError(_("speed of circular trajectory must be positive"))
        if not num_circles > 0:
            raise ValueError(_("number of circles must be positive"))

        self.radius = radius
        self.speed = speed
        self.clockwise = clockwise
        self.start_angle_rad = np.radians(start_angle)

        # We'll use the circumference to do much of the calculations.
        self._circumference = 2 * np.pi * radius
        self._length = num_circles * self._circumference
        self._duration = self._length / speed

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
        # Angle from centre of circle to position of system. 0° is along the x axis and
        # 90° along the y axis.
        t = np.asarray(t)
        rel_angle = 2 * np.pi * t * self.speed / self._circumference
        if self.clockwise:
            angle = self.start_angle_rad + rel_angle
        else:
            angle = self.start_angle_rad - rel_angle

        # Corresponding system position.
        pos = np.full(t.shape + (3,), self.centre, dtype=float)
        pos[:, 0] += self.radius * np.cos(angle)
        pos[:, 1] += self.radius * np.sin(angle)

        # Mask out invalid times.
        invalid = (t < 0) | (t > self._duration)
        pos[invalid] = np.nan
        return pos

    def orientation(self, t: ArrayLike) -> quaternionic.QArray:
        # Heading of the system. 0° is pointing along the x axis and 90° along y.
        t = np.asarray(t)
        rel_heading = 2 * np.pi * t * self.speed / self._circumference
        if self.clockwise:
            heading = self.start_angle_rad + np.pi / 2 + rel_heading
        else:
            heading = self.start_angle_rad - np.pi / 2 - rel_heading

        # Components of the quaternion.
        ori = np.zeros(t.shape + (4,), dtype=float)
        ori[:, 0] = np.cos(heading / 2)
        ori[:, 3] = np.sin(heading / 2)

        # Mask out invalid times.
        invalid = (t < 0) | (t > self._duration)
        ori[invalid] = np.nan

        return quaternionic.array(ori)

    def velocity(self, t: ArrayLike) -> np.ndarray:
        # Heading of the system. 0° is pointing along the x axis and 90° along y.
        t = np.asarray(t)
        rel_heading = 2 * np.pi * t * self.speed / self._circumference
        if self.clockwise:
            heading = self.start_angle_rad + np.pi / 2 + rel_heading
        else:
            heading = self.start_angle_rad - np.pi / 2 - rel_heading

        # Corresponding velocity.
        vel = np.zeros(t.shape + (3,), dtype=float)
        vel[:, 0] = self.speed * np.cos(heading)
        vel[:, 1] = self.speed * np.sin(heading)

        # Mask out invalid times.
        invalid = (t < 0) | (t > self._duration)
        vel[invalid] = np.nan
        return vel


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
        start_position
            The start position of the trajectory in the global coordinate system.
        end_position
            The end position of the trajectory in the global coordinate system.
        speed
            The speed of the system in metres per second.
        start_time
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

        # Calculate the orientation of the system. We store the result as a NumPy array
        # here as quaternionic arrays cannot be pickled for transfer between workers.
        self._ori = quaternion_from_vectors([1, 0, 0], diff).ndarray

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

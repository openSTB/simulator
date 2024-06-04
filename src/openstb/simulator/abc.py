# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Abstract base classes for the openSTB simulator.

These classes define the expected interface for plugins. Note that it is not required
for plugin classes to derive from an abstract base class; the simulator does not check
for this at runtime, and so any class which meets the specification will work.

Having said that, deriving from a base class has some benefits. If an attempt is made to
instantiate an incomplete class (e.g., if you forgot to implement a method), an
exception will be raised immediately. This may be easier and less frustrating to debug
than an error which occurs during the simulation. Deriving from a base class also allows
static type checkers (such as the mypy checker used on the core code) to do more
detailed checking.

"""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike
import quaternionic


class Plugin(ABC):
    pass


class Trajectory(Plugin):
    """The trajectory followed by the sonar."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """The duration of the trajectory in seconds."""
        pass

    @property
    @abstractmethod
    def length(self) -> float:
        """The length of the trajectory in seconds."""
        pass

    @property
    @abstractmethod
    def start_time(self) -> datetime:
        """The UTC time of the first sample in the trajectory.

        Note that this can be generated at first access (e.g., set to the current date
        and time), but this must return the same value for all subsequent accesses.

        """
        pass

    @abstractmethod
    def position(self, t: ArrayLike) -> np.ndarray:
        """Calculate the position of the sonar at a given time.

        Parameters
        ----------
        t : array-like of floats
            The times of interest in seconds relative to the first sample of the
            trajectory.

        Returns
        -------
        position : numpy.ndarray
            The position of the sonar at each requested time. This will have a shape
            (..., 3) where ``...`` is the shape of the input ``t``. Values where the
            given time is less than zero or greater than the trajectory's duration will
            be set to `np.nan`.

        """
        pass

    @abstractmethod
    def orientation(self, t: ArrayLike) -> quaternionic.QArray:
        """Calculate the orientation of the sonar at a given time.

        Parameters
        ----------
        t : array-like of floats
            The times of interest in seconds relative to the first sample of the
            trajectory.

        Returns
        -------
        orientation : numpy.ndarray
            The orientation of the sonar at each requested time as quaternions
            representing the rotation of the global x axis to the vehicle's x axis. This
            will have a shape (..., 4) where ``...`` is the shape of the input ``t``.
            Values where the given time is less than zero or greater than the
            trajectory's duration will be set to `np.nan`.

        """
        pass

    @abstractmethod
    def velocity(self, t: ArrayLike) -> np.ndarray:
        """Calculate the velocity of the sonar at a given time.

        Parameters
        ----------
        t : array-like of floats
            The times of interest in seconds relative to the first sample of the
            trajectory.

        Returns
        -------
        velocity : numpy.ndarray
            The velocity of the sonar at each requested time. This will have a shape
            (..., 3) where ``...`` is the shape of the input ``t``. Values where the
            given time is less than zero or greater than the trajectory's duration will
            be set to `np.nan`.

        """
        pass

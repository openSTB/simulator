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
from types import TracebackType
from typing import overload

import dask.distributed
import numpy as np
from numpy.typing import ArrayLike
import quaternionic


class Plugin(ABC):
    pass


class Beampattern(Plugin):
    """The far-field beampattern of an aperture."""

    @abstractmethod
    def evaluate(self, wavelength: ArrayLike, direction: ArrayLike) -> np.ndarray:
        """Evaluate the beampattern.

        Parameters
        ----------
        wavelength : array-like
            The wavelengths to evaluate at.
        direction : array-like
            The unit direction vectors to evaluate for. The last dimension must have
            size 3 and contain the x, y and z components of the vector, with the x axis
            being the normal of the transducer, the y axis being the 'width' direction
            and the z axis being the vertical direction. Excluding the last dimension,
            this must be broadcastable with ``wavelength``.

        Returns
        -------
        amplitude : numpy.array
            The normalised amplitude of the beampattern. This will have the broadcast
            shape of the inputs. Directions behind the aperture (i.e., where the x
            component of ``direction`` is negative) may be set to zero, or may be
            evaluated.

        """
        pass


class Cluster(Plugin):
    """Interface to a Dask cluster to perform computations.

    A Cluster plugin is responsible for configuring Dask to use the desired computing
    environment, whether that is a collection of processes on the local computer or
    within a high performance computing (HPC) system.

    Note that this base class implements a context manager which calls `initialise` on
    entry (returning the Client to use) and `terminate` on exit. This should be
    acceptable for most plugins, but can be overridden if needed.

    """

    @abstractmethod
    def initialise(self) -> dask.distributed.Client:
        """Initialise the cluster for use.

        Returns
        -------
        client : dask.distributed.Client
            The client interface used to submit tasks to the cluster.

        """
        pass

    def terminate(self):
        """Terminate use of the cluster.

        The default implementation does nothing. Plugins should implement this function
        if needed to cleanly stop the cluster.

        """
        pass

    def __enter__(self) -> dask.distributed.Client:
        return self.initialise()

    @overload
    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None: ...

    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.terminate()


class Signal(Plugin):
    """The signal transmitted by the sonar."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """The duration of the signal in seconds."""

    @abstractmethod
    def sample(self, t: ArrayLike, baseband_frequency: float) -> np.ndarray:
        """Sample the signal in the baseband.

        Parameters
        ----------
        t : array-like
            The times (in seconds) to sample the signal at with t=0 corresponding to the
            start of transmission.
        baseband_frequency : float
            The reference frequency (in Hz) used to downconvert the signal to the
            analytic baseband.

        Returns
        -------
        samples : numpy.ndarray
            Complex baseband samples of the signal. Values corresponding to times before
            transmission starts (t < 0) or after transmission finished (t > duration)
            should be set to zero.

        """
        pass


class SignalWindow(Plugin):
    """A window which can be applied to a signal."""

    @abstractmethod
    def get_samples(
        self, t: ArrayLike, duration: float, fill_value: float = 0
    ) -> np.ndarray:
        """Get the samples of the window.

        Parameters
        ----------
        t : array-like
            The times (in seconds) to get window samples for, with t=0 corresponding to
            the start of the window.
        duration : float
            The length of the window in seconds.
        fill_value : float
            The value to use for samples outside the window.

        Returns
        -------
        samples : numpy.ndarray
            The samples of the window as an array of floats. Values outside the window
            should be set to ``fill_value``.

        """
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


class PingTimes(Plugin):
    """Times that the sonar starts transmitting a ping."""

    @abstractmethod
    def calculate(self, trajectory: Trajectory) -> np.ndarray:
        """Calculate the ping times.

        Parameters
        ----------
        trajectory : openstb.simulator.abc.Trajectory
            The trajectory being followed by the system.

        Returns
        -------
        ping_times : numpy.ndarray
            A one-dimensional array of floats giving the ping start times in seconds
            since the start of the trajectory.

        """
        pass

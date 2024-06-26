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
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import overload

import dask.distributed
import numpy as np
from numpy.typing import ArrayLike
import quaternionic


class Plugin(ABC):
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


class Environment(Plugin):
    """Properties of the environment the sonar is operating in."""

    @abstractmethod
    def salinity(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        """Get the salinity of the water.

        Parameters
        ----------
        t : array-like
            The times of interest in seconds since the start of the trajectory.
        position : array-like
            The positions of interest in global coordinates. This must have a final axis
            of size 3 containing the coordinates. All other axes must be broadcastable
            with ``t``.

        Returns
        -------
        salinity : numpy.ndarray
            The salinity of the water in parts per thousand at the requested times and
            positions. This will be compatible with the broadcast shape of ``t`` and
            ``position`` (ignoring the final axis of ``position``). Some axes may have
            size one if the salinity is constant along them.

        """
        pass

    @abstractmethod
    def sound_speed(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        """Get the speed of sound.

        Parameters
        ----------
        t : array-like
            The times of interest in seconds since the start of the trajectory.
        position : array-like
            The positions of interest in global coordinates. This must have a final axis
            of size 3 containing the coordinates. All other axes must be broadcastable
            with ``t``.

        Returns
        -------
        sound_speed : numpy.ndarray
            The speed of sound in metres per second at the requested times and
            positions. This will be compatible with the broadcast shape of ``t`` and
            ``position`` (ignoring the final axis of ``position``). Some axes may have
            size one if the sound speed is constant along them.

        """
        pass

    @abstractmethod
    def temperature(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        """Get the temperature of the water.

        Parameters
        ----------
        t : array-like
            The times of interest in seconds since the start of the trajectory.
        position : array-like
            The positions of interest in global coordinates. This must have a final axis
            of size 3 containing the coordinates. All other axes must be broadcastable
            with ``t``.

        Returns
        -------
        sound_speed : numpy.ndarray
            The temperature in degrees Celsius at the requested times and positions.
            This will be compatible with the broadcast shape of ``t`` and ``position``
            (ignoring the final axis of ``position``). Some axes may have size one if
            the temperature is constant along them.

        """
        pass


class Signal(Plugin):
    """The signal transmitted by the sonar."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """The duration of the signal in seconds."""

    @property
    @abstractmethod
    def minimum_frequency(self) -> float:
        """The minimum frequency of the signal in Hertz."""

    @property
    @abstractmethod
    def maximum_frequency(self) -> float:
        """The minimum frequency of the signal in Hertz."""

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


class Target(Plugin):
    """A target in the scene being simulated.

    The initialiser of a target plugin should only perform basic error checking. At the
    point when it is initialised, there may be other plugins not yet initialised that
    will subsequently fail initialisation. Expensive computations should be placed in
    the `prepare` method which will only be called once all plugins have successfully
    been initialised.

    Note that any properties or methods defined by a plugin may not be usable until
    its `prepare` method has been run.

    """

    def prepare(self) -> None:
        """Prepare the target for simulation."""
        pass


class PointTargets(Target):
    """An object made up of simple point targets.

    Note that these targets have no directional or material information, and so only
    simple scattering with a constant amplitude scaling of the incident pulse is
    possible.

    """

    @abstractmethod
    def __len__(self) -> int:
        """The number of point targets."""
        pass

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """An Nx3 array of the position of each point target."""
        pass

    @property
    @abstractmethod
    def reflectivity(self) -> np.ndarray:
        """An Nx1 array of the amplitude scale factor used to model scattering."""
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


@dataclass(slots=True, eq=False, order=False)
class TravelTimeResult:
    """Results of a travel time calculation.

    Note that the arrays must have the expected number of dimensions, but any of the
    axes may be length 1 if the results are repeated along that axis. For example,
    `rx_position` may have shape (N_receivers, 1, 3) if the receivers are in the same
    position for all echoes.

    """

    #: An array of shape (N_receivers, N_targets) containing the two-way travel time in
    #: seconds for the pulse to travel from the transmitter to each target and then back
    #: to each target.
    travel_time: np.ndarray

    #: An array of shape (3,) with the position of the transmitter in global coordinates
    #: when the transmission started.
    tx_position: np.ndarray

    #: A quaternion array of shape (4,) with the orientation of the transmitter in the
    #: global system when the transmission started.
    tx_orientation: quaternionic.QArray

    #: An array of shape (3,) with velocity vectors of the sonar in the global system
    #: at the time the transmission started.
    tx_velocity: np.ndarray

    #: An array of shape (N_targets, 3) with unit vectors in the global coordinate
    #: system for the direction the pulse left the transmitter at to reach each target.
    tx_vector: np.ndarray

    #: An array of shape (N_targets,) with the total path length, in metres, that the
    #: pulse followed from the transmitter to each target.
    tx_path_length: np.ndarray

    #: An array of shape (N_receivers, N_targets, 3) with the position of the receivers
    #: in global coordinates when the echoes reached the receivers.
    rx_position: np.ndarray

    #: A quaternion array of shape (N_receivers, N_targets, 4) with the orientation of
    #: the receivers in the global system when the echoes reached them.
    rx_orientation: quaternionic.QArray

    #: An array of shape (N_receivers, N_targets, 3) with velocity vectors of the sonar
    #: in the global system at the time the echoes were received.
    rx_velocity: np.ndarray

    #: An array of shape (N_receivers, N_targets, 3) with unit vectors in the global
    #: coordinate system for the direction the echo from each target was travelling when
    #: it reached each receiver.
    rx_vector: np.ndarray

    #: An array of shape (N_receivers, N_targets) with the total path length, in metres,
    #: that the echo from each target took to reach each receiver.
    rx_path_length: np.ndarray

    #: An array of shape (N_receivers, N_targets) with multiplicative scale factors to
    #: apply to the amplitude of the echo from each target, e.g., due to attenuation or
    #: geometric spreading loss. If None, no scale factors will be applied.
    scale_factor: np.ndarray | None = None


class TravelTime(Plugin):
    """Calculates the time taken for a pulse to travel to a target and back."""

    @abstractmethod
    def calculate(
        self,
        trajectory: Trajectory,
        ping_time: float,
        environment: Environment,
        tx_position: ArrayLike,
        tx_orientation: ArrayLike | quaternionic.QArray,
        rx_positions: ArrayLike,
        rx_orientations: ArrayLike | quaternionic.QArray,
        target_positions: ArrayLike,
    ) -> TravelTimeResult:
        """Calculate the two-way travel time.

        Parameters
        ----------
        trajectory : openstb.simulator.abc.Trajectory
            A trajectory plugin instance representing the trajectory followed by the
            system carrying the sonar.
        ping_time : float
            The time, in seconds relative to the start of the trajectory, that the ping
            transmission was started.
        tx_position : array-like
            The position of the transmitter in the vehicle coordinate system. This must
            be a vector of length 3 containing the x, y and z components of the
            position.
        rx_positions : array-like
            The positions of each receiver in the vehicle coordinate system. This must
            be an array of shape (Nr, 3) containing the x, y and z components of the
            position for all Nr receivers.
        target_positions : array-like
            The position of each target in the global coordinate system. This must be an
            array of shape (Nt, 3) containing the x, y and z components of all Nt
            targets.

        Returns
        -------
        result : TravelTimeResult
            The result of the calculation.

        """
        pass


class ScaleFactor(Plugin):
    """A multiplicative scale factor applied to the amplitude of an echo."""

    @abstractmethod
    def calculate(
        self,
        ping_time: float,
        f: np.ndarray,
        environment: Environment,
        signal_frequency_bounds: tuple[float, float],
        tt_result: TravelTimeResult,
    ) -> np.ndarray:
        """Calculate the scale factor to apply.

        Parameters
        ----------
        ping_time : float
            The time, in seconds relative to the start of the trajectory, that the ping
            transmission was started.
        f : numpy.ndarray
            The frequencies the simulation is being performed at.
        environment : openstb.simulator.abc.Environment
            Parameters of the environment the system is operating in.
        signal_frequency_bounds : tuple
            A tuple of floats (minimum frequency, maximum frequency) giving the
            frequency bounds of the transmitted signal.
        tt_result : openstb.simulator.abc.TravelTimeResult
            The results of the travel time calculations for this piece of the
            simulation.

        Returns
        -------
        scale_factor : numpy.ndarray
            Multiplicative scale factors to apply to the echoes. This should have a
            shape (N_receiver, N_frequencies, N_targets) where N_receiver and N_targets
            are the number of receivers and targets, respectively, in the travel time
            results, and N_frequencies is the number of frequencies in ``f``. Any of
            the axes may be length 1 if the scale factor is constant along that axis.

        """
        pass

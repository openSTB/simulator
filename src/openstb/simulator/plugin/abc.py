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
from enum import Enum
from typing import Any, Generic

import dask.distributed
import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.types import SimulationConfig

_ = translations.load("openstb.simulator").gettext


class Plugin(ABC):
    pass


class Simulation(Plugin, Generic[SimulationConfig]):
    """A type of simulation."""

    @abstractmethod
    def run(self, config: SimulationConfig):
        """Run the simulation.

        Parameters
        ----------
        config : mapping
            A mapping configuring the simulation. The specific structure will be defined
            by each plugin.

        """
        pass

    @property
    @abstractmethod
    def config_class(src) -> type[SimulationConfig]:
        """The simulation class for this type of simulation.

        Note that this is the class, not an instance of the class.

        """
        pass


class ConfigLoader(Plugin):
    """A plugin which can load simulation configuration from somewhere."""

    @abstractmethod
    def load(self) -> dict:
        """Load the configuration.

        Returns
        -------
        dict
            A dictionary containing the configuration.

        """
        pass

    @classmethod
    @abstractmethod
    def could_handle(cls, source: str) -> bool:
        """Guess if this plugin could load a given configuration source.

        This is intended for use in user interfaces where a configuration source, such
        as a filename, is given by the user. This method should guess if it would be
        able to load a configuration from the source. This check should be quick and
        without side-effects; false positives and false negatives are acceptable. In the
        former case, the subsequent attempt to load will result in an error and the user
        being told to specify a config loader plugin. If no plugins return True, then
        the user will again be told to specify a plugin.

        Parameters
        ----------
        source : str
            The provided configuration source.

        Returns
        -------
        bool

        """
        pass


class DaskCluster(Plugin):
    """Interface to a Dask cluster to perform computations.

    A DaskCluster plugin is responsible for configuring Dask to use the desired
    computing environment, whether that is a collection of processes on the local
    computer or within a high performance computing (HPC) system.

    """

    @abstractmethod
    def initialise(self):
        """Initialise the cluster for use.

        This must be able to be called multiple times without problems.

        """
        pass

    @classmethod
    def initialise_worker(cls) -> bool:
        """Initialise a worker for this cluster.

        This is intended for cluster environments where each worker runs in a separate
        process, such as MPI, and some configuration or connection parameters need to be
        passed to workers. This class method should arrange for this to happen, and for
        the worker processes to join the cluster.

        It is not required for plugins to implement this method if it is not appropriate
        for their environment.

        Returns
        -------
        boolean
            If True, the caller is the main simulation controller process and should
            proceed with the simulation on return. If False, the caller was a worker
            process and should exit on return.

        """
        raise NotImplementedError(
            _("this cluster does not support separate worker initialisation")
        )

    @property
    @abstractmethod
    def client(self) -> dask.distributed.Client:
        """Get a Client to use the cluster with.

        This should raise an exception if the cluster has not been initialised.

        """
        pass

    def terminate(self):
        """Terminate use of the cluster.

        This must be able to be called on an uninitialised cluster or a
        previously-terminated cluster.

        The default implementation does nothing. Plugins should implement this function
        if needed to cleanly stop the cluster. Note that there is no guarantee that this
        function will be called; if a clean-up step is required, the plugin should
        implement `__del__` or use something like `weakref.finalize` to ensure this is
        run before the instance is garbage collected.

        """
        pass


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
        """The length of the trajectory in metres."""
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


class Distortion(Plugin):
    """An effect which distorts the echo signal."""

    @abstractmethod
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
        """
        Parameters
        ----------
        ping_time : float
            The time, in seconds relative to the start of the trajectory, that the ping
            transmission was started.
        f : array-like
            A one-dimensional array of the frequencies (in Hertz) that the simulation is
            being performed at. Note that these are the real frequencies, but the
            simulation is performed in the complex baseband.
        S : array-like
            The Fourier coefficients of the current echo signal in the complex baseband.
            This will be a three-dimensional array with dimensions (receiver, f,
            target). The first and last dimensions may have size 1 if there is currently
            no variation in the signal over that dimension.
        baseband_frequency : float
            The carrier frequency used to shift the signal into the baseband.
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
        modified_S : numpy.ndarray
            The complex Fourier coefficients of the modified signals at the frequencies
            in ``f``. This should be the same dimensions as the input ``S``, potentially
            with the ``receiver`` and ``target`` dimensions expanded to full size if the
            distortion is receiver- and/or target-dependent.

        """
        pass


class ResultFormat(Enum):
    """Standard formats the simulation type plugins may use for storing results."""

    #: A Zarr group for baseband pressure simulations. This has variables sample_time,
    #: ping_start_time and pressure, and attributes baseband_frequency and sample_rate.
    ZARR_BASEBAND_PRESSURE = 1


class ResultConverter(Plugin):
    """Convert a simulator result from its internal format to a desired format."""

    @abstractmethod
    def can_handle(self, format: ResultFormat | str, config: SimulationConfig) -> bool:
        """Check if this plugin will be able to convert a simulation result.

        Parameters
        ----------
        format : ResultFormat, str
            The format of the simulation result. Simulation type plugins provided by the
            main package will use one of the values from the `ResultFormat` enum. Those
            from external plugins may still use a standard format, or may use a string
            to refer to a custom format.
        config : SimulationConfig
            The simulation configuration. The `SimulationConfigt` type represents a
            mapping with string keys that will vary based on the type of simulation
            being run.

        Returns
        -------
        Boolean
            If this plugin expects to be able to convert the simulation result to its
            desired final format.

        """
        pass

    @abstractmethod
    def convert(
        self, format: ResultFormat | str, result: Any, config: SimulationConfig
    ) -> bool:
        """Convert a simulation result.

        Parameters
        ----------
        format : ResultFormat, str
            The format of the simulation result. Simulation types provided by the main
            package will use one of the values from the `ResultFormat` enum. Simulation
            types from external plugins may still use a standard format, or may use a
            string to refer to a custom format.
        result
            The simulation result. Simulation type plugins provided by the main package
            will use a `zarr.Group` instance. Other plugins may use different structures
            to hold the result.
        config : SimulationConfig
            The simulation configuration. The `SimulationConfig` type represents a
            mapping with string keys that will vary based on the type of simulation
            being run.

        Returns
        -------
        success : Boolean
            True if the result was successfully converted. The simulator may choose to
            delete the initial result in this case. False if the conversion failed and
            the initial result should be retained.

        """
        pass


class Transducer(Plugin):
    """A single transducer within the sonar system."""

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """The position of the transducer in the vehicle coordinate system.

        The origin of the vehicle coordinate system is the point at which the trajectory
        is recorded. The x axis points forward, the y axis to starboard and the z axis
        down.

        Returns
        -------
        numpy.ndarray
            A 3-element vector containing the position.

        """
        pass

    @property
    @abstractmethod
    def orientation(self) -> quaternionic.QArray:
        """The orientation of the transducer in the vehicle coordinate system.

        The origin of the vehicle coordinate system is the point at which the trajectory
        is recorded. The x axis points forward, the y axis to starboard and the z axis
        down.

        An unrotated transducer has a normal parallel to the x axis. Rotating with this
        quaternion gives the orientation of the transducer.

        Returns
        -------
        quaternionic.array
            A single quaternion giving the orientation.

        """
        pass

    @property
    def distortion(self) -> list[Distortion]:
        """Echo signal distortions associated with the transducer.

        Returns
        -------
        list of Distortion instances

        """
        return []


class System(Plugin):
    """A collection of other plugins representing a sonar system.

    Note that any of the properties may be None. The meaning of None can be decided by
    the simulation type, but would typically indicate that the corresponding plugin
    should be directly specified in the configuration instead of through the system.

    """

    @property
    @abstractmethod
    def transmitter(self) -> Transducer | None:
        """The transmitting transducer for the system.

        Returns
        -------
        Transducer

        """
        pass

    @property
    @abstractmethod
    def receivers(self) -> list[Transducer] | None:
        """The receiving transducers for the system.

        Returns
        -------
        list of Transducer instances

        """
        pass

    @property
    @abstractmethod
    def signal(self) -> Signal | None:
        """The signal transmitted by the system.

        Returns
        -------
        Signal

        """
        pass

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Simple simulation with idealised point targets."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from typing import NotRequired, TypedDict

from dask.tokenize import tokenize
import distributed
import numpy as np
import zarr

from openstb.i18n.support import translations
from openstb.simulator.controller.base import LoopController
from openstb.simulator.plugin import abc
from openstb.simulator.target.points import target_chunk_iterator
from openstb.simulator.utils.reduction import DaskReductionTree

_ = translations.load("openstb.simulator").gettext


class SimplePointConfig(TypedDict):
    """Specification for the SimplePointSimulation configuration dictionary."""

    dask_cluster: abc.DaskCluster
    """Dask cluster to run the simulation on."""

    echo_distortion: NotRequired[list[abc.Distortion]]
    """Plugins to apply distortion to the echoed signal.

    These are applied to the signal after it has been scattered by a target and before
    it reaches the receiver. They are applied in the order they are given here.

    """

    emitted_distortion: NotRequired[list[abc.Distortion]]
    """Plugins to apply distortion to the emitted signal.

    These are applied to the signal after it has been transmitted and before it reaches
    the target. They are applied in the order they are given here.

    """

    environment: abc.Environment
    """Plugin to provide details about the simulated environment."""

    ping_times: abc.PingTimes
    """Plugin to calculate ping start times."""

    result_converter: NotRequired[abc.ResultConverter]
    """Plugin to convert the simulation output into a desired format."""

    system: abc.System
    """Details about the system to simulate."""

    targets: list[abc.PointTargets]
    """A list of plugins giving the point targets to simulate."""

    trajectory: abc.Trajectory
    """Plugin specifying the trajectory followed by the system."""

    travel_time: abc.TravelTime
    """Plugin to calculate the travel times to and from each target."""


@dataclass(slots=True, eq=False, order=False)
class CommonSettings:
    """Container for simulation settings that are common to all chunks.

    These settings are broadcast to all workers before starting the simulation as they
    do not change during the simulation.

    """

    f: np.ndarray
    """Passband simulation frequencies."""

    S: np.ndarray
    """Baseband spectrum of the transmitted signal."""

    signal_frequency_bounds: tuple[float, float]
    """Passband frequency bounds of the transmitted signal."""

    baseband_frequency: float
    """Frequency used for basebanding."""

    travel_time: abc.TravelTime
    """Travel time calculator to use."""

    trajectory: abc.Trajectory
    """Trajectory followed by the system."""

    environment: abc.Environment
    """Environmental properties to simulate."""

    tx_position: np.ndarray
    """Position of the transmitter in system coordinates."""

    tx_ori: np.ndarray
    """Orientation of the transmitter in system coordinates."""

    emitted_distortion: list[abc.Distortion]
    """Distortions to apply to the signal before it hits a target.

    This should include any distortions from the transmitter.

    """

    echo_distortion: list[abc.Distortion]
    """Distortions to apply to the echo before it reaches the receiver."""


@dataclass(slots=True, eq=False, order=False)
class ChunkSettings:
    """Container for simulation settings specific to a single chunk.

    These settings vary per ping or per receiver, and so have to be set for each chunk.

    """

    ping_time: float
    """Time the ping starts in seconds relative to the trajectory start."""

    rx_position: np.ndarray
    """Position of the receiver in system coordinates."""

    rx_ori: np.ndarray
    """Orientation of the receiver in system coordinates."""

    rx_distortion: list[abc.Distortion]
    """Receiver distortions to apply to the received signal."""

    max_t: float
    """Maximum sample time in seconds relative to the start of transmission."""


def point_simulation_chunk(
    chunk: tuple[int, int, np.ndarray, np.ndarray],
    params: tuple[CommonSettings, ChunkSettings],
) -> np.ndarray:
    """Perform a single chunk of the simulation.

    Parameters
    ----------
    chunk
        The chunk of targets to simulate. This corresponds to one item of the iterator
        return by [openstb.simulator.target.points.target_chunk_iterator][].
    params
        A tuple of the common and chunk-specific settings.

    Returns
    -------
    echo_fdomain
        The frequeny-domain echo for this chunk of the simulation.

    """
    # Unpack the inputs.
    _, _, position, reflectivity = chunk
    common, settings = params

    # Calculate the travel times.
    tt_result = common.travel_time.calculate(
        common.trajectory,
        settings.ping_time,
        common.environment,
        common.tx_position,
        common.tx_ori,
        settings.rx_position.reshape(1, 3),
        settings.rx_ori.reshape(1, 4),
        position,
    )

    # Start with the transmitted spectrum.
    Schunk = common.S[np.newaxis, :, np.newaxis]

    # Apply any distortions from the transmitter or during its travel to the target.
    for distortion in common.emitted_distortion:
        Schunk = distortion.apply(
            settings.ping_time,
            common.f,
            Schunk,
            common.baseband_frequency,
            common.environment,
            common.signal_frequency_bounds,
            tt_result,
        )

    # Apply the phase shift corresponding to each travel time.
    Schunk = Schunk * np.exp(
        -2j * np.pi * common.f[:, np.newaxis] * tt_result.travel_time[:, np.newaxis, :]
    )
    Schunk *= (tt_result.travel_time <= settings.max_t)[:, np.newaxis, :]

    # Scale by the reflectivity of the target.
    Schunk *= reflectivity

    # Apply any distortions from its travel during return or by the receiver.
    for distortion in common.echo_distortion + settings.rx_distortion:
        Schunk = distortion.apply(
            settings.ping_time,
            common.f,
            Schunk,
            common.baseband_frequency,
            common.environment,
            common.signal_frequency_bounds,
            tt_result,
        )

    # Sum over the targets.
    return Schunk.sum(axis=-1).squeeze()


def point_simulation_store(
    storage: zarr.Array, ping: int, receiver: int, echo_fdomain: np.ndarray
) -> None:
    """Store the result of a piece of the simulation.

    Parameters
    ----------
    storage
        The zarr array to store the result in.
    ping
        The ping index of the result.
    receiver
        The receiver of the result.
    echo_fdomain
        The Fourier domain echoes to add to the storage. This is returned to the time
        domain and added to the current result in the storage.

    """
    # Return to the time domain.
    result = np.fft.ifft(np.fft.ifftshift(echo_fdomain, axes=-1), norm="forward")

    # Remove any guard band.
    Nt = storage.shape[-1]
    if Nt < result.shape[-1]:
        result = result[..., :Nt]

    if Nt > result.shape[-1]:
        raise ValueError(_("result has fewer samples than expected"))

    with distributed.Lock("write-pressure"):
        storage[ping, receiver, :] += result  # type:ignore[operator]


class SimplePointSimulation(
    LoopController[
        tuple[int, int, np.ndarray, np.ndarray],  # Chunk type
        tuple[CommonSettings, ChunkSettings],  # Parameter type
        np.ndarray,  # Result type
        SimplePointConfig,  # Configuration type
    ]
):
    """Controller plugin to simulate with idealised point targets.

    The echo from each point target is summed to get the final result. Occlusions are
    not modelled (the targets are infinitesimally small and so cannot cast shadows).
    The scattering strength of each target is a fixed value independent of aspect. The
    simulation is performed in the temporal frequency domain and the results are
    basebanded.

    The targets are divided into chunks and submitted to the cluster for simulation. To
    combine the results from multiple chunks, a reduction tree is used. This sums the
    results from a small number of chunks, allowing the initial results to be freed from
    memory. Groups of these summed results are themselves recursively summed in the same
    manner until a single combined result remains. For eight chunks and a reduction node
    count of two, this means instead of computing the result as

        result = r1 + r2 + r3 + r4 + r5 + r6 + f7 + r8

    we might compute it as

        result = ((r1 + r2) + (r3 + r4)) + ((r5 + r6) + (r7 + r8))

    which has the same number of operations but allows memory to be freed earlier. Note
    that, since floating-point operations are generally not associative, these two
    results will differ by some small amount proportional to the machine precision and
    the number of operations.

    This reduction will be performed for a few levels, after which the combined set of
    results will be written to disk. The number of chunks in each write is given by
    the `reduction_node_count` parameter raised to the power of the `reduction_levels`
    parameter.

    """

    def __init__(
        self,
        result_filename: os.PathLike[str] | str,
        points_per_chunk: int,
        sample_rate: float,
        baseband_frequency: float,
        max_samples: int | None = None,
        task_lower_threshold: float = 2.0,
        task_upper_threshold: float = 3.0,
        reduction_node_count: int = 4,
        reduction_levels: int = 3,
    ):
        """
        Parameters
        ----------
        result_filename
            Filename to store the results under. If this already exists, an exception
            will be raised.
        points_per_chunk
            The maximum number of point targets to simulate in each chunk.
        sample_rate
            Sampling rate in Hertz of the results.
        baseband_frequency
            Frequency used for downconversion during basebanding (carrier frequency).
        max_samples
            The maximum number of samples each receiver will capture per ping. If not
            given, this is calculated from the maximum interval between pings and the
            sampling rate. The maximum length of the trace in seconds can be found by
            dividing `max_samples` by `sample_rate`.
        task_lower_threshold
            When the number of simulation tasks per worker in the scheduler drops below
            this value, add more tasks.
        task_upper_threshold
            When submitting simulation tasks to the scheduler, add enough to ensure
            there are at least this many per worker.
        reduction_node_count
            How many results to sum at each level of the reduction tree.
        reduction_levels
            How many levels to use in the reduction tree before writing the combined
            results to disk.

        """
        super().__init__()

        # Do not overwrite existing results.
        self.result_filename = Path(result_filename)
        if self.result_filename.exists():
            raise ValueError(_("specified output path already exists"))

        # Basic parameter checks.
        if points_per_chunk < 1:
            raise ValueError(_("points per chunk must be at least one"))
        if sample_rate < 1:
            raise ValueError(_("sample rate must be at least one"))
        if baseband_frequency < 0:
            raise ValueError(_("baseband frequency cannot be negative"))
        if reduction_node_count < 2:
            raise ValueError(_("reduction node count must be at least two"))
        if reduction_levels < 1:
            raise ValueError(_("reduction levels must be at least one"))

        # If given, max samples must be given.
        if max_samples is not None and max_samples < 1:
            raise ValueError(_("max samples must be at least one"))

        # Ensure thresholds are valid.
        if task_lower_threshold < 1:
            raise ValueError(_("task lower threshold must be at least one"))
        if task_upper_threshold < task_lower_threshold:
            raise ValueError(
                _("task upper threshold cannot be less than lower threshold")
            )

        self.points_per_chunk = points_per_chunk
        self.sample_rate = sample_rate
        self.baseband_frequency = baseband_frequency
        self.max_samples = max_samples
        self.loop_lower_threshold = task_lower_threshold
        self.loop_upper_threshold = task_upper_threshold
        self.reduction_node_count = reduction_node_count
        self.reduction_levels = reduction_levels

    @property
    def config_class(self):
        return SimplePointConfig

    def run(self, config: SimplePointConfig):
        logger = logging.getLogger(__name__)
        logger.info(_("Preparing for point target simulation"))

        # Ensure any result converter will be able to work.
        result_format = abc.ResultFormat.ZARR_BASEBAND_PRESSURE
        if "result_converter" in config:
            logger.info(_("Checking result converter is suitable"))
            if not config["result_converter"].can_handle(result_format, config):
                raise ValueError(_("output converter cannot handle results"))

        # Determine the number of receivers being simulated.
        Nr = len(config["system"].receivers)

        # Calculate the ping start times.
        ping_start = config["ping_times"].calculate(config["trajectory"])
        Np = len(ping_start)
        if Np == 0:
            raise ValueError(_("no pings in the simulation"))

        # Calculate the maximum number of samples we will simulate.
        if self.max_samples is None:
            if Np == 1:
                raise ValueError(
                    _("max_samples must be specified for a single-ping simulation")
                )
            max_interval = np.diff(ping_start).max()
            Ns = int(np.ceil(max_interval * self.sample_rate))
        else:
            Ns = self.max_samples

        # We will add a guard band during the simulation. This means the echo of a
        # target close to the end of the maximum range will not wrap round to the start
        # of the trace. The samples in the guard band will be discarded before the trace
        # is saved. We add a factor of 1.1 to account for distortions.
        signal = config["system"].signal
        gb_size = int(np.ceil(signal.duration * 1.1 * self.sample_rate))

        # Calculate the corresponding sample times and evaluate the signal.
        t = np.arange(Ns + gb_size) / self.sample_rate
        s = signal.sample(t, self.baseband_frequency)
        signal_frequency_bounds = (
            signal.minimum_frequency,
            signal.maximum_frequency,
        )

        # Prepare the targets.
        N_points = 0
        for target in config["targets"]:
            target.prepare()
            N_points += len(target)
        if N_points == 0:
            raise ValueError(_("no targets to simulate"))

        # Prepare the output storage. We checked it was non-existent in __init__,
        # but check again in case the path has been created in the meantime.
        if self.result_filename.exists():
            raise ValueError(_("specified output path already exists"))
        local_store = zarr.storage.LocalStore(self.result_filename)
        storage = zarr.create_group(store=local_store)

        # Use a default value of zero for the pressure. This does not write zero to the
        # chunks, but sets it as a default if accessed (e.g., to add a result).
        pressure = storage.zeros(
            name="pressure", shape=(Np, Nr, Ns), chunks=(1, 1, Ns), dtype="c16"
        )

        # Add the sample time and ping times.
        st = storage.empty(name="sample_time", shape=(Ns,), chunks=(Ns,), dtype="f8")
        st[:] = t[:Ns]
        pst = storage.empty(
            name="ping_start_time", shape=(Np,), chunks=(Np,), dtype="f8"
        )
        pst[:] = ping_start

        # Add some metadata.
        storage.attrs["baseband_frequency"] = self.baseband_frequency
        storage.attrs["sample_rate"] = self.sample_rate

        logger.info(
            _(
                "Simulation size: %(Np)d pings, %(Nr)d receivers, %(Ns)d samples per "
                "trace, %(Nt)d point targets"
            ),
            {"Np": Np, "Nr": Nr, "Ns": Ns, "Nt": N_points},
        )

        # The bulk of the simulation is carried out in the frequency domain.
        f = np.fft.fftshift(np.fft.fftfreq(Ns + gb_size, 1 / self.sample_rate))
        S = np.fft.fftshift(np.fft.fft(s, norm="forward"))

        # Prepare the cluster.
        logger.info(_("Initialising Dask cluster"))
        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

        # Collate settings that are common for every chunk of the simulation and
        # broadcast it to all workers.
        logger.info(_("Sending common details to cluster workers"))
        transmitter = config["system"].transmitter
        emitted_distortion = transmitter.distortion + config.get(
            "emitted_distortion", []
        )
        echo_distortion = config.get("echo_distortion", [])
        common = client.scatter(
            CommonSettings(
                f=f + self.baseband_frequency,
                S=S,
                baseband_frequency=self.baseband_frequency,
                travel_time=config["travel_time"],
                trajectory=config["trajectory"],
                tx_position=transmitter.position,
                tx_ori=transmitter.orientation.ndarray,
                environment=config["environment"],
                emitted_distortion=emitted_distortion,
                echo_distortion=echo_distortion,
                signal_frequency_bounds=signal_frequency_bounds,
            ),
            broadcast=True,
        )

        # Set the simulation function to use.
        self.loop_simulate = point_simulation_chunk

        # Create a function to get the target chunks for a given ping and receiver.
        def loop_chunks(loop_key: tuple[int, int]):
            return target_chunk_iterator(config["targets"], self.points_per_chunk)

        self.loop_chunks = loop_chunks

        # And another to get the simulation parameters.
        def loop_params(loop_key: tuple[int, int]):
            nonlocal common

            p, r = loop_key
            return common, ChunkSettings(
                ping_time=ping_start[p],
                rx_position=config["system"].receivers[r].position,
                rx_ori=config["system"].receivers[r].orientation.ndarray,
                rx_distortion=config["system"].receivers[r].distortion,
                max_t=Ns / self.sample_rate,
            )

        self.loop_params = loop_params

        # Set the log message to be output at the start of each iterations.
        self.loop_log_message = "Starting submission of tasks for ping %0d receiver %0d"

        # Create a function to add a result to the Zarr storage. Note that this needs to
        # add the future to the loop_futures set so that the simulation loop tracks its
        # progress and doesn't finish until the store is complete.
        def store(result: distributed.Future, loop_key: tuple[int, int]):
            self.loop_futures.add(
                client.submit(
                    point_simulation_store,
                    pressure,
                    loop_key[0],
                    loop_key[1],
                    result,
                    key=f"store-{tokenize(result, loop_key)}",
                )
            )

        # Create a reduction tree. This will be given a list of results. As a NumPy
        # array, the list corresponds to axis 0 so sum over that axis.
        rtree = DaskReductionTree(
            client,
            store,
            np.sum,
            reduce_kwargs={"axis": 0},
            futures=self.reduction_node_count,
            levels=self.reduction_levels,
        )

        # And run the simulation loop for all pings and receivers
        logger.info(_("Beginning simulation"))
        self.loop_run(client, (Np, Nr), rtree)

        # The simulation has completed. All loop futures have been completed, so we can
        # remove the common data from the workers.
        del common
        logger.info(_("Simulation complete"))

        # And try to convert the format if desired.
        if "result_converter" in config:
            logger.info(_("Passing results to result converter"))

            success = config["result_converter"].convert(
                abc.ResultFormat.ZARR_BASEBAND_PRESSURE, storage, config
            )
            local_store.close()

            if success:
                shutil.rmtree(self.result_filename)
            else:
                logger.error(
                    _(
                        "Result conversion failed. Original simulation results can be "
                        "found at %s"
                    ),
                    self.result_filename,
                )

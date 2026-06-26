# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Iterator
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
class _CommonSettings:
    """Container for simulation settings that are common to all chunks."""

    # Simulation frequencies (passband).
    f: np.ndarray

    # Spectrum of the transmitted signal (baseband).
    S: np.ndarray

    # Frequency bounds of the transmitted signal (passband).
    signal_frequency_bounds: tuple[float, float]

    # Frequency used for basebanding.
    baseband_frequency: float

    # Travel time calculator.
    travel_time: abc.TravelTime

    # Trajectory followed by the system.
    trajectory: abc.Trajectory

    # Environment details.
    environment: abc.Environment

    # Position of the transmitter in the system.
    tx_position: np.ndarray

    # Orientation of the transmitter in the system.
    tx_ori: np.ndarray

    # Distortion of the signal prior to incidence.
    emitted_distortion: list[abc.Distortion]

    # Distortions to apply to the echo before it reaches the receiver.
    echo_distortion: list[abc.Distortion]


def _point_simulation_chunk(
    position: np.ndarray,
    reflectivity: np.ndarray,
    ping_time: float,
    rx_position: np.ndarray,
    rx_ori: np.ndarray,
    rx_distortion: list[abc.Distortion],
    common: _CommonSettings,
    max_t: float,
) -> np.ndarray:
    """Simulation of a chunk of receivers and/or targets."""
    # Calculate the travel times.
    tt_result = common.travel_time.calculate(
        common.trajectory,
        ping_time,
        common.environment,
        common.tx_position,
        common.tx_ori,
        rx_position.reshape(1, 3),
        rx_ori.reshape(1, 4),
        position,
    )

    # Start with the transmitted spectrum.
    Schunk = common.S[np.newaxis, :, np.newaxis]

    # Apply any distortions from the transmitter or during its travel to the target.
    for distortion in common.emitted_distortion:
        Schunk = distortion.apply(
            ping_time,
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
    Schunk *= (tt_result.travel_time <= max_t)[:, np.newaxis, :]

    # Scale by the reflectivity of the target.
    Schunk *= reflectivity

    # Apply any distortions from its travel during return or by the receiver.
    for distortion in common.echo_distortion + rx_distortion:
        Schunk = distortion.apply(
            ping_time,
            common.f,
            Schunk,
            common.baseband_frequency,
            common.environment,
            common.signal_frequency_bounds,
            tt_result,
        )

    # Sum over the targets.
    return Schunk.sum(axis=-1).squeeze()


def _point_simulation_store(
    storage: zarr.Array, ping: int, receiver: int, result_fdomain: np.ndarray
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
    result_fdomain
        The Fourier domain result to add to the storage. This are returned to the time
        domain and added to the current result in the storage.

    """
    # Return to the time domain.
    result = np.fft.ifft(np.fft.ifftshift(result_fdomain), norm="forward")

    # Remove any guard band.
    Nt = storage.shape[-1]
    if Nt < result.shape[-1]:
        result = result[..., :Nt]

    if Nt > result.shape[-1]:
        raise ValueError(_("result has fewer samples than expected"))

    with distributed.Lock("write-pressure"):
        storage[ping, receiver, :] += result  # type:ignore[operator]


class SimplePointSimulation(abc.Controller[SimplePointConfig]):
    """Simulation using idealised point targets.

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
        self.task_lower_threshold = task_lower_threshold
        self.task_upper_threshold = task_upper_threshold
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

        # Create a function which is called when a chunk of results has been reduced and
        # schedules the reduced values to be added to the pressure variable.
        def store_reduced(future: distributed.Future) -> None:
            key = f"store-result-{self.ping}-{self.receiver}-{tokenize(future)}"
            self.tasks.add(
                client.submit(
                    _point_simulation_store,
                    pressure,
                    self.ping,
                    self.receiver,
                    future,
                    key=key,
                )
            )

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
            _CommonSettings(
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

        # Turn the submission thresholds into integers.
        Nworkers = len(client.scheduler_info(n_workers=-1)["workers"])
        lower_threshold = int(np.ceil(Nworkers * self.task_lower_threshold))
        upper_threshold = int(np.ceil(Nworkers * self.task_upper_threshold))

        # Initialise our state variables.
        self.tasks: set[distributed.Future] = set()
        self.ping = 0
        self.receiver = 0
        self.target_iter: None | Iterator = None
        self.reduction = DaskReductionTree(
            client,
            store_reduced,
            np.sum,
            reduce_kwargs={"axis": 0},
            levels=self.reduction_levels,
            futures=self.reduction_node_count,
        )
        self.sim_tasks = 0

        logger.info(_("Beginning simulation"))

        # Submit the first set of tasks.
        self._submit_tasks(
            upper_threshold,
            client,
            config,
            ping_start=ping_start,
            store_var=pressure,
            max_t=Ns / self.sample_rate,
            common=common,
        )

        # And loop until all tasks are complete.
        while self.tasks:
            # If the number of tasks has dropped under our lower threshold, add more.
            if self.sim_tasks < lower_threshold:
                # Update the number of workers (they may be added to or removed from the
                # cluster dynamically). We don't need to do this every loop; every time
                # we think we need to add tasks should be often enough.
                Nworkers = len(client.scheduler_info(n_workers=-1)["workers"])
                lower_threshold = int(np.ceil(Nworkers * self.task_lower_threshold))
                upper_threshold = int(np.ceil(Nworkers * self.task_upper_threshold))

                to_add = upper_threshold - self.sim_tasks
                if to_add > 0:
                    self._submit_tasks(
                        to_add,
                        client,
                        config,
                        ping_start=ping_start,
                        store_var=pressure,
                        max_t=Ns / self.sample_rate,
                        common=common,
                    )

            # Wait for something to complete.
            res = distributed.wait(self.tasks, return_when="FIRST_COMPLETED")
            self.tasks = res.not_done

            # Check for failures in the completed tasks.
            for future in res.done:
                if future.key.startswith("simulate-"):
                    self.sim_tasks -= 1

                if future.status == "error":
                    raise future.exception()
                if future.status == "cancelled":
                    raise RuntimeError(_("a future was cancelled"))

            del res
            del future

        # The simulation has completed. Remove the common data.
        del common
        logger.info(_("Simulation complete"))

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

    def _submit_tasks(
        self,
        count: int,
        client: distributed.Client,
        config: SimplePointConfig,
        ping_start: np.ndarray,
        store_var: zarr.Array,
        **sim_kwargs,
    ):
        """Submit simulation tasks to a cluster.

        Parameters
        ----------
        count
            The number of simulation tasks to add.
        client
            The Dask client to submit tasks to.
        config
            The simulation configuration.
        ping_start
            The start time of each ping.
        store_var
            The Zarr variable to store results under.

        """
        logger = logging.getLogger(__name__)

        # No more pings left to simulate.
        if self.ping == -1:
            return

        # Have completed a ping+receiver, reset the target iterator for the next.
        if self.target_iter is None:
            self.target_iter = target_chunk_iterator(
                config["targets"], self.points_per_chunk
            )
            logger.info(
                _("Starting submission of tasks for ping %(P)d receiver %(R)d"),
                {"P": self.ping, "R": self.receiver},
            )

        # Add the simulation tasks.
        rx = config["system"].receivers[self.receiver]
        t = ping_start[self.ping]
        for n in range(count):
            try:
                target_idx, chunk_idx, pos, refl = next(self.target_iter)
            except StopIteration:
                self.target_iter = None
                break

            future = client.submit(
                _point_simulation_chunk,
                pos,
                refl,
                key=f"simulate-{target_idx}-{chunk_idx}-{self.ping}-{self.receiver}",
                ping_time=t,
                rx_position=rx.position,
                rx_ori=rx.orientation.ndarray,
                rx_distortion=rx.distortion,
                **sim_kwargs,
            )
            self.tasks.add(future)
            self.sim_tasks += 1
            self.reduction.add_futures(future)

        # We have finished this ping+receiver pair.
        if self.target_iter is None:
            # Ensure the final tasks are summed and saved.
            self.reduction.flush()

            # Move on to the next ping+receiver to simulate.
            self.receiver += 1
            if self.receiver >= len(config["system"].receivers):
                self.receiver = 0
                self.ping += 1
                if self.ping >= len(ping_start):
                    self.ping = -1

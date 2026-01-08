# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Iterator
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from typing import Any, MutableMapping, NotRequired, TypedDict, cast

from dask.tokenize import tokenize
import distributed
import numpy as np
import zarr

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc
from openstb.simulator.plugin.util import flatten_system

_ = translations.load("openstb.simulator").gettext


class SimplePointConfig(TypedDict):
    """Specification for the SimplePointSimulation configuration dictionary."""

    # Dask cluster to run the simulation on.
    dask_cluster: abc.DaskCluster

    #: Plugin which will calculate ping start times.
    ping_times: abc.PingTimes

    #: System information.
    system: NotRequired[abc.System]

    #: Transducer used for transmitting the signal.
    transmitter: NotRequired[abc.Transducer]

    #: Plugin representing the transmitted signal.
    signal: NotRequired[abc.Signal]

    #: Transducers for which the received signal should be simulated.
    receivers: NotRequired[list[abc.Transducer]]

    #: A list of plugins giving the point targets to simulate.
    targets: list[abc.PointTargets]

    #: Plugin specifying the trajectory followed by the system.
    trajectory: abc.Trajectory

    #: Details about the environment the system is operating in.
    environment: abc.Environment

    #: Plugin which will calculate the travel times to and from each target.
    travel_time: abc.TravelTime

    #: Plugins which will apply distortions to each echo.
    distortion: NotRequired[list[abc.Distortion]]

    #: Plugin to convert the output into a desired format.
    result_converter: NotRequired[abc.ResultConverter]


@dataclass(slots=True, eq=False, order=False)
class CommonSettings:
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

    # Distortion of the signal.
    distortion: list[abc.Distortion]


def _target_chunk_iterator(
    targets: list[abc.PointTargets],
    points_per_chunk: int,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Iterator over all available target chunks.

    Parameters
    ----------
    targets
        The targets to simulate.
    points_per_chunk
        How many points to simulate in each chunk.

    Returns
    -------
    target_idx : int
        The zero-based index of the target in the input list.
    chunk_idx : int
        The zero-based index (within the original target) of the first point in the
        chunk.
    position : np.ndarray
        An (N, 3) array of the position of the points in the chunk.
    reflectivity : np.ndarray
        An (N,) array of the reflectivity of the points in the chunk.

    """
    for idx, target in enumerate(targets):
        N = len(target)
        for n in range(0, N, points_per_chunk):
            if (n + points_per_chunk) < N:
                count = points_per_chunk
            else:
                count = -1

            yield idx, n, *target.get_chunk(n, count)


def _point_simulation_chunk(
    position: np.ndarray,
    reflectivity: np.ndarray,
    ping_time: float,
    rx_position: np.ndarray,
    rx_ori: np.ndarray,
    rx_distortion: list[abc.Distortion],
    common: CommonSettings,
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

    # Distort.
    for distortion in common.distortion:
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

    # Apply each receive distortion.
    for distortion in rx_distortion:
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
    storage: zarr.Array, ping: int, receiver: int, results_fdomain: list[np.ndarray]
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
    results_fdomain
        The Fourier domain results to add to the storage. These are summed, returned to
        the time domain and added to the current result in the storage.

    """
    # Combine the results, noting we may have a single-element list.
    if len(results_fdomain) == 1:
        result_fdomain = results_fdomain[0]
    else:
        result_fdomain = np.sum(results_fdomain, axis=0)

    # Return to the time domain.
    result = np.fft.ifft(np.fft.ifftshift(result_fdomain))

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
    results will differ by some small amount proportional to the machine precision.

    This reduction will be performed for a few levels, after which the combined set of
    results will be written to disk. The number of chunks in each write is given by
    reduction_node_count ** reduction_levels.

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

        flatten_system(cast(MutableMapping[str, Any], config))

        # Ensure any result converter will be able to work.
        result_format = abc.ResultFormat.ZARR_BASEBAND_PRESSURE
        if "result_converter" in config:
            logger.info(_("Checking result converter is suitable"))
            if not config["result_converter"].can_handle(result_format, config):
                raise ValueError(_("output converter cannot handle results"))

        # Determine the number of receivers being simulated.
        Nr = len(config["receivers"])

        # Calculate the ping start times.
        ping_start = config["ping_times"].calculate(config["trajectory"])
        Np = len(ping_start)
        if Np == 0:
            raise ValueError(_("no pings in the simulation"))

        # Calculate the maximum number of samples we will simulate.
        if self.max_samples is None:
            max_interval = np.diff(ping_start).max()
            Ns = int(np.ceil(max_interval * self.sample_rate))
        else:
            Ns = self.max_samples

        # We will add a guard band during the simulation. This means the echo of a
        # target close to the end of the maximum range will not wrap round to the start
        # of the trace. The samples in the guard band will be discarded before the trace
        # is saved. We add a factor of 1.1 to account for distortions.
        gb_size = int(np.ceil(config["signal"].duration * 1.1 * self.sample_rate))

        # Calculate the corresponding sample times and evaluate the signal.
        t = np.arange(Ns + gb_size) / self.sample_rate
        s = config["signal"].sample(t, self.baseband_frequency)
        signal_frequency_bounds = (
            config["signal"].minimum_frequency,
            config["signal"].maximum_frequency,
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
        S = np.fft.fftshift(np.fft.fft(s))

        # Prepare the cluster.
        logger.info(_("Initialising Dask cluster"))
        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

        # Collate settings that are common for every chunk of the simulation and
        # broadcast it to all workers.
        logger.info(_("Sending common details to cluster workers"))
        distortion = config["transmitter"].distortion + config.get("distortion", [])
        common = client.scatter(
            CommonSettings(
                f=f + self.baseband_frequency,
                S=S,
                baseband_frequency=self.baseband_frequency,
                travel_time=config["travel_time"],
                trajectory=config["trajectory"],
                tx_position=config["transmitter"].position,
                tx_ori=config["transmitter"].orientation.ndarray,
                environment=config["environment"],
                distortion=distortion,
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
        self.to_reduce: list[list[distributed.Future]] = []
        for i in range(self.reduction_levels):
            self.to_reduce.append([])
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
            self.target_iter = _target_chunk_iterator(
                config["targets"], self.points_per_chunk
            )
            logger.info(
                _("Starting submission of tasks for ping %(P)d receiver %(R)d"),
                {"P": self.ping, "R": self.receiver},
            )

        # Add the simulation tasks.
        rx = config["receivers"][self.receiver]
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

            self._reduce_and_store(client, store_var, future)

        # We have finished this ping+receiver pair.
        if self.target_iter is None:
            # Ensure the final tasks are summed and saved.
            self._reduce_and_store(client, store_var, None)

            # Move on to the next ping+receiver to simulate.
            self.receiver += 1
            if self.receiver >= len(config["receivers"]):
                self.receiver = 0
                self.ping += 1
                if self.ping >= len(ping_start):
                    self.ping = -1

    def _reduce_and_store(
        self,
        client: distributed.Client,
        store_var: zarr.Array,
        new_task: distributed.Future | None,
    ):
        """Generation reduction tree and store tasks.

        Parameters
        ----------
        client
            The Dask client to submit tasks to.
        store_var
            The array to store results in.
        new_task
            A new simulation task to add to a reduction tree. If None, sum all tasks
            currently in the reduction tree and store. This can be used to flush the
            tree at the end of the simulation tasks for a ping+receiver pair.

        """
        future = new_task

        for i in range(self.reduction_levels):
            # Store any given future at this level.
            if future is not None:
                self.to_reduce[i].append(future)

            # If we are still adding tasks, check if we have enough futures at this
            # level to sum up and move to the next.
            if new_task is not None:
                if len(self.to_reduce[i]) < self.reduction_node_count:
                    break

            # There are no more tasks for this ping. Check if there are any remaining
            # futures at this level.
            else:
                if not self.to_reduce[i]:
                    future = None
                    continue

            # Sum all results at this level and store if this is the final level.
            inputs = list(self.to_reduce[i])
            self.to_reduce[i].clear()
            if i == (self.reduction_levels - 1):
                key = f"store-result-{self.ping}-{self.receiver}-{tokenize(future)}"
                self.tasks.add(
                    client.submit(
                        _point_simulation_store,
                        store_var,
                        self.ping,
                        self.receiver,
                        inputs,
                        priority=i,
                        key=key,
                    )
                )

            else:
                # At the end of the ping+receiver pair, we may have a single result
                # which we can pass directly on to the next level.
                if len(inputs) > 1:
                    key = f"reducing-sum-{tokenize(inputs)}"
                    future = client.submit(np.sum, inputs, axis=0, key=key, priority=i)
                    self.tasks.add(future)
                else:
                    future = inputs[0]

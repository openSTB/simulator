# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from typing import Any, MutableMapping, NotRequired, TypedDict, cast

from dask.tokenize import tokenize
import distributed
import numpy as np
from numpy.typing import NDArray
import zarr

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc
from openstb.simulator.plugin.util import flatten_system

_ = translations.load("openstb.simulator").gettext


class PointSimulationConfig(TypedDict):
    """Specification for the PointSimulator configuration dictionary."""

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


def _point_simulation_chunk(
    common: CommonSettings,
    targets: np.ndarray,
    ping_time: float,
    rx_position: np.ndarray,
    rx_ori: np.ndarray,
    rx_distortion: list[abc.Distortion],
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
        targets[:, :3],
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
    Schunk *= targets[:, 3]

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
    storage: zarr.Array,
    result_fdomain: np.ndarray,
    guard_band_size: int,
    ping: int,
    receiver: int | slice,
    update: bool = False,
) -> None:
    """Store the result of a piece of the simulation.

    Parameters
    ----------
    result_fdomain : numpy.ndarray
        The Fourier domain result of the simulation.
    storage : zarr.Array
        The zarr array to store the result in.
    ping : int
        The ping index of the result.
    receiver : int, slice
        The receiver index or indices of the result.
    update : Boolean
        If True, add the result to the existing value; this will use a lock to prevent
        race issues. If False, assume only one result per index and just store it. In
        the latter case, it is also assumed that the chunk size of the storage matches
        the chunk size of the simulation so it is safe for multiple chunks to be
        simultaneously written.

    """
    # Return to the time domain and remove the guard band.
    result = np.fft.ifft(np.fft.ifftshift(result_fdomain))[..., :-guard_band_size]

    if update:
        with distributed.Lock("write-pressure"):
            storage[ping, receiver, :] += result
    else:
        storage[ping, receiver, :] = result


class PointSimulation(abc.Simulation[PointSimulationConfig]):
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
    size of two, this means instead of computing the result as

        result = r1 + r2 + r3 + r4 + r5 + r6 + f7 + r8

    we compute it as

        result = ((r1 + r2) + (r3 + r4)) + ((r5 + r6) + (r7 + r8))

    which has the same number of operations but allows memory to be freed earlier. Note
    that, since floating-point operations are generally not associative, these two
    results will differ by some small amount proportional to the machine precision.

    """

    def __init__(
        self,
        result_filename: os.PathLike[str] | str,
        targets_per_chunk: int,
        sample_rate: float,
        baseband_frequency: float,
        max_samples: int | None = None,
        task_threshold: float = 2.0,
        reduction_node_size: int = 3,
    ):
        """
        Parameters
        ----------
        result_filename : path-like
            Filename to store the results under. If this already exists, an exception
            will be raised.
        targets_per_chunk : int
            The maximum number of targets to simulate in each chunk.
        sample_rate : float
            Sampling rate in Hertz of the results.
        baseband_frequency : float
            Frequency used for downconversion during basebanding (carrier frequency).
        max_sample : int, optional
            The maximum number of samples each receiver will capture per ping. If not
            given, this is calculated from the maximum interval between pings and the
            sampling rate.
        task_threshold : float
            Ensure that there are at least this many simulation tasks for each worker in
            the scheduler. This does not include task to sum the results from different
            chunks or to store the results.
        reduction_node_size : int
            How many

        """
        # Do not overwrite existing results.
        self.result_filename = Path(result_filename)
        if self.result_filename.exists():
            raise ValueError(_("specified output path already exists"))

        self.targets_per_chunk = targets_per_chunk
        self.sample_rate = sample_rate
        self.baseband_frequency = baseband_frequency
        self.max_samples = max_samples
        self.task_threshold = task_threshold
        self.reduction_node_size = reduction_node_size

    @property
    def config_class(self):
        return PointSimulationConfig

    def _submit(
        self,
        client: distributed.Client,
        config: PointSimulationConfig,
        common: CommonSettings,
        ping: int,
        ping_time: float,
        receiver: int,
        max_t: float,
        targets: Sequence[distributed.Future | NDArray],
        reduction_node_size: int,
    ) -> tuple[list[distributed.Future], distributed.Future]:
        sim_futures = [
            client.submit(
                _point_simulation_chunk,
                common,
                targets=chunk,
                ping_time=ping_time,
                rx_position=config["receivers"][receiver].position,
                rx_ori=config["receivers"][receiver].orientation.ndarray,
                rx_distortion=config["receivers"][receiver].distortion,
                max_t=max_t,
            )
            for chunk in targets
        ]

        # No reduction tree; sum all pieces at once.
        if reduction_node_size <= 1:
            key = f"result-sum-{tokenize(sim_futures)}"
            summed = client.submit(np.sum, sim_futures, axis=0, key=key)
            return sim_futures, summed

        # Generate a sum reduction tree with each branch combining N chunks.
        tree_futures = sim_futures
        while len(tree_futures) > 1:
            new_futures = []
            for i in range(0, len(tree_futures), reduction_node_size):
                tmp = tree_futures[i : i + reduction_node_size]
                if len(tmp) == 1:
                    new_futures.append(tmp[0])
                else:
                    key = f"reducing-sum-{tokenize(tmp)}"
                    new_futures.append(client.submit(np.sum, tmp, axis=0, key=key))
                del tmp
            tree_futures = new_futures

        return sim_futures, tree_futures[0]

    def run(self, config: PointSimulationConfig):
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

        # Add a guard band. This means the echo of a target close to the end of the
        # maximum range will not wrap round to the start of the trace.
        gb_size = int(np.ceil(config["signal"].duration * 1.1 * self.sample_rate))

        # Calculate the corresponding sample times and evaluate the signal.
        t = np.arange(Ns + gb_size) / self.sample_rate
        s = config["signal"].sample(t, self.baseband_frequency)
        signal_frequency_bounds = (
            config["signal"].minimum_frequency,
            config["signal"].maximum_frequency,
        )

        # The bulk of the simulation is carried out in the frequency domain.
        f = np.fft.fftshift(np.fft.fftfreq(Ns + gb_size, 1 / self.sample_rate))
        S = np.fft.fftshift(np.fft.fft(s))

        # Prepare the targets.
        N_targets = 0
        for target in config["targets"]:
            target.prepare()
            N_targets += len(target)
        if N_targets == 0:
            raise ValueError(_("no targets to simulate"))

        logger.info(
            _(
                "Simulation size: %(Np)d pings, %(Nr)d receivers, %(Ns)d samples per "
                "trace, %(Nt)d targets"
            ),
            {"Np": Np, "Nr": Nr, "Ns": Ns, "Nt": N_targets},
        )

        # Combine into an array of position and reflectivity.
        targets = np.empty((N_targets, 4), dtype=float)
        start = 0
        for target in config["targets"]:
            targets[start : start + len(target), :3] = target.position
            targets[start : start + len(target), 3] = target.reflectivity
            start += len(target)

        # Prepare the cluster.
        logger.info(_("Initialising Dask cluster"))
        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

        # Split the targets into chunks and distribute.
        N_chunks = int(np.ceil(N_targets / self.targets_per_chunk))
        chunks = np.array_split(targets, N_chunks)
        chunks = client.scatter(chunks, broadcast=False)

        # Collate and send common details to all workers.
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
        st[:] = t
        pst = storage.empty(
            name="ping_start_time", shape=(Np,), chunks=(Np,), dtype="f8"
        )
        pst[:] = ping_start

        # Add some metadata.
        storage.attrs["baseband_frequency"] = self.baseband_frequency
        storage.attrs["sample_rate"] = self.sample_rate

        # Futures for simulation tasks that have not yet completed.
        sim_futures: set[distributed.Future] = set()

        # Futures for result store tasks that have not yet completed.
        store_futures: set[distributed.Future] = set()

        logger.info(_("Beginning simulation"))
        for ping in range(Np):
            logger.info(_("Starting submission of jobs for ping %(N)d"), {"N": ping})
            for receiver in range(Nr):
                # Number of simulation tasks we should aim for. We update this here in
                # case workers are added to or removed from the cluster.
                Nworkers = len(client.scheduler_info()["workers"])
                threshold = Nworkers * self.task_threshold

                # Wait until the number of simulation futures are below this.
                while len(sim_futures) > threshold:
                    res = distributed.wait(sim_futures, return_when="FIRST_COMPLETED")
                    sim_futures = set(res.not_done)

                    # Drop references to any store commands that have completed. This
                    # allows the scheduler to remove them and the tasks that they
                    # depended on.
                    store_futures = {f for f in store_futures if not f.done()}

                # Add simulation and reduction tasks for this ping-receiver pair.
                sim, reduced = self._submit(
                    client,
                    config,
                    common,
                    ping,
                    ping_start[ping],
                    receiver,
                    Ns / self.sample_rate,
                    chunks,
                    reduction_node_size=self.reduction_node_size,
                )
                sim_futures.update(sim)

                # Add a task to store the reduced result.
                store = client.submit(
                    _point_simulation_store,
                    pressure,
                    reduced,
                    gb_size,
                    ping,
                    receiver,
                    key=f"store-result-{ping}-{receiver}",
                )
                store_futures.add(store)

                # We don't want to keep these references locally.
                del reduced
                del store

        # Remove our references to the data and settings we scattered at the start. The
        # scheduler can then remove them when no tasks using them remain.
        del chunks
        del common

        # Tasks for all pings and receivers have been submitted. Wait until the results
        # have been written to disk.
        distributed.wait(store_futures)
        del store_futures
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

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, MutableMapping, NotRequired, TypedDict, cast

import distributed
import numpy as np
import zarr

from openstb.i18n.support import domain_translator
from openstb.simulator.plugin import abc
from openstb.simulator.plugin.util import flatten_system

_ = domain_translator("openstb.simulator", plural=False)


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
class _ChunkCommon:
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
    common: _ChunkCommon,
    targets: np.ndarray,
    ping_time: float,
    rx_position: np.ndarray,
    rx_ori: np.ndarray,
    rx_distortion: list[abc.Distortion],
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
    result_fdomain: np.ndarray,
    storage: zarr.Array,
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
    # Return to the time domain.
    result = np.fft.ifft(np.fft.ifftshift(result_fdomain))

    if update:
        with distributed.Lock("write-pressure"):
            storage[ping, receiver, :] += result
    else:
        storage[ping, receiver, :] = result


class PointSimulation(abc.Simulation[PointSimulationConfig]):
    def __init__(
        self,
        result_filename: os.PathLike[str] | str,
        targets_per_chunk: int,
        sample_rate: float,
        baseband_frequency: float,
        max_samples: int | None = None,
    ):
        # Do not overwrite existing results.
        self.result_filename = Path(result_filename)
        if self.result_filename.exists():
            raise ValueError(_("specified output path already exists"))

        self.targets_per_chunk = targets_per_chunk
        self.sample_rate = sample_rate
        self.baseband_frequency = baseband_frequency
        self.max_samples = max_samples

    @property
    def config_class(self):
        return PointSimulationConfig

    def run(self, config: PointSimulationConfig):
        flatten_system(cast(MutableMapping[str, Any], config))

        config["dask_cluster"].initialise()
        client = config["dask_cluster"].client

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

        # Calculate the corresponding sample times and evaluate the signal.
        t = np.arange(Ns) / self.sample_rate
        s = config["signal"].sample(t, self.baseband_frequency)
        signal_frequency_bounds = (
            config["signal"].minimum_frequency,
            config["signal"].maximum_frequency,
        )

        # The bulk of the simulation is carried out in the frequency domain.
        f = np.fft.fftshift(np.fft.fftfreq(Ns, 1 / self.sample_rate))
        S = np.fft.fftshift(np.fft.fft(s))

        if "result_converter" in config:
            if not config["result_converter"].can_handle(
                abc.ResultFormat.ZARR_BASEBAND_PRESSURE, config
            ):
                raise ValueError(_("output converter cannot handle results"))

        # Prepare the targets.
        N_targets = 0
        for target in config["targets"]:
            target.prepare()
            N_targets += len(target)
        if N_targets == 0:
            raise ValueError(_("no targets to simulate"))

        # Combine into an array of position and reflectivity.
        targets = np.empty((N_targets, 4), dtype=float)
        start = 0
        for target in config["targets"]:
            targets[start : start + len(target), :3] = target.position
            targets[start : start + len(target), 3] = target.reflectivity
            start += len(target)

        # Split the targets into chunks and distribute.
        N_chunks = int(np.ceil(N_targets / self.targets_per_chunk))
        chunks = np.array_split(targets, N_chunks)
        chunks = client.scatter(chunks, broadcast=False)

        # Collate and send common details to all workers.
        distortion = config["transmitter"].distortion + config["distortion"]
        common = client.scatter(
            _ChunkCommon(
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
        store = zarr.storage.LocalStore(self.result_filename)
        storage = zarr.create_group(store=store)

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

        for ping in range(Np):
            for receiver in range(Nr):
                # Simulate each chunk of targets.
                chunk_futures = [
                    client.submit(
                        _point_simulation_chunk,
                        common,
                        targets=chunk,
                        ping_time=ping_start[ping],
                        rx_position=config["receivers"][receiver].position,
                        rx_ori=config["receivers"][receiver].orientation.ndarray,
                        rx_distortion=config["receivers"][receiver].distortion,
                    )
                    for chunk in chunks
                ]

                # To avoid keeping all results in memory, generate a reduction tree
                # which successively sums groups of intermediate results until we
                # get a final result.
                N_per_sum = 3
                while len(chunk_futures) > 1:
                    new_futures = []
                    for i in range(0, len(chunk_futures), N_per_sum):
                        tmp = chunk_futures[i : i + N_per_sum]
                        if len(tmp) == 1:
                            new_futures.append(tmp[0])
                        else:
                            new_futures.append(client.submit(np.sum, tmp, axis=0))
                        del tmp
                    chunk_futures = new_futures
                    del new_futures

                # Then store the result.
                finish = client.submit(
                    _point_simulation_store, chunk_futures[0], pressure, ping, receiver
                )
                del chunk_futures
                distributed.wait(finish)
                del finish

        # Remove references to the data we scattered at the start. Although this
        # will go out of scope when we exit the context manager, the cluster may not
        # have processed this when the context manager exiting triggers a shutdown.
        # The shutdown sees the data still on the workers and issues warnings about
        # a loss of computed tasks.
        del chunks
        del common

        if "result_converter" in config:
            success = config["result_converter"].convert(
                abc.ResultFormat.ZARR_BASEBAND_PRESSURE, storage, config
            )
            store.close()
            if success:
                shutil.rmtree(self.result_filename)

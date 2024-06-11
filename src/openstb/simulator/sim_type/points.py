# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
from typing import TypedDict

import distributed
import numpy as np
from numpy.typing import ArrayLike
import zarr

from openstb.i18n.support import domain_translator
from openstb.simulator import abc


_ = domain_translator("openstb.simulator", plural=False)


class PointSimulatorConfigDict(TypedDict):
    """Specification for the PointSimulator configuration dictionary."""

    #: Plugin which will calculate ping start times.
    ping_times: abc.PingTimes

    #: Position of each receiver in the vehicle coordinate system. Must be an Nr x 3
    #: array where Nr is the number of receivers.
    receiver_position: ArrayLike

    #: Plugin representing the transmitted signal.
    signal: abc.Signal

    #: A list of plugins giving the point targets to simulate.
    targets: list[abc.PointTargets]

    #: Plugin specifying the trajectory followed by the system.
    trajectory: abc.Trajectory

    #: Position of the transmitter in the vehicle coordinate system. Must be an array of
    #: shape (3,) i.e., only a single transmitter.
    transmitter_position: ArrayLike

    #: Plugin which will calculate the travel times to and from each target.
    travel_time: abc.TravelTime


def _pointsim_chunk(
    f: np.ndarray,
    S: np.ndarray,
    target_pos: np.ndarray,
    travel_time: abc.TravelTime,
    trajectory: abc.Trajectory,
    ping_time: float,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
):

    tt_result = travel_time.calculate(
        trajectory, ping_time, tx_position, rx_position.reshape(1, 3), target_pos
    )

    Schunk = S[:, np.newaxis] * np.exp(
        -2j * np.pi * f[:, np.newaxis] * tt_result.travel_time[:, np.newaxis, :]
    )

    return Schunk.sum(axis=-1).squeeze()


class PointSimulator:
    def __init__(
        self,
        output_filename: os.PathLike[str] | str,
        targets_per_chunk: int,
        sample_rate: float,
        baseband_frequency: float,
        max_samples: int | None = None,
        fill_value: complex | str = "nan",
    ):
        self.output_filename = Path(output_filename)
        self.targets_per_chunk = targets_per_chunk
        self.sample_rate = sample_rate
        self.baseband_frequency = baseband_frequency
        self.max_samples = max_samples
        if isinstance(fill_value, str):
            if fill_value == "nan":
                self.fill_value = np.nan + 0j
            else:
                raise ValueError(
                    _("unsupported fill_value '{value}'").format(value=fill_value)
                )
        else:
            self.fill_value = fill_value

    def run(self, client: distributed.Client, config: PointSimulatorConfigDict):
        # Check the shapes of the transmitter and receiver info.
        tx_position = np.array(config["transmitter_position"], dtype=float)
        if tx_position.shape != (3,):
            raise ValueError(_("invalid shape for transmitter position array"))

        rx_position = np.array(config["receiver_position"])
        if rx_position.ndim != 2 or rx_position.shape[-1] != 3:
            raise ValueError(_("invalid shape for receiver position array"))
        Nr = rx_position.shape[0]

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

        # The bulk of the simulation is carried out in the frequency domain.
        f = np.fft.fftshift(np.fft.fftfreq(Ns, 1 / self.sample_rate))
        S = np.fft.fftshift(np.fft.fft(s))

        # Prepare the targets.
        N_targets = 0
        for target in config["targets"]:
            target.prepare()
            N_targets += len(target)
        if N_targets == 0:
            raise ValueError(_("no targets to simulate"))

        N_chunks = int(np.ceil(N_targets / self.targets_per_chunk))
        chunks = np.array_split(config["targets"][0].position, N_chunks)
        chunks = client.scatter(chunks, broadcast=False)

        # Send common details to all workers.
        chunk_args = dict(
            f=client.scatter(f, broadcast=True),
            S=client.scatter(S, broadcast=True),
            travel_time=client.scatter(config["travel_time"], broadcast=True),
            trajectory=client.scatter(config["trajectory"], broadcast=True),
            tx_position=client.scatter(tx_position, broadcast=True),
        )

        # Prepare the output storage.
        if self.output_filename.exists():
            import shutil

            shutil.rmtree(self.output_filename)

        store = zarr.DirectoryStore(self.output_filename)
        storage = zarr.group(store=store)
        storage.create_dataset(
            "results", shape=(Np, Nr, Ns), chunks=(1, 1, Ns), dtype="c16"
        )
        storage.create_dataset("sample_time", shape=(Ns,), chunks=(Ns,), dtype="f8")
        storage["sample_time"][:] = t

        for p in range(Np):
            for r in range(Nr):
                # Simulate each chunk of targets.
                chunk_futures = [
                    client.submit(
                        _pointsim_chunk,
                        ping_time=ping_start[p],
                        rx_position=rx_position[r],
                        target_pos=chunk,
                        **chunk_args,
                    )
                    for chunk in chunks
                ]

                # To avoid keeping all results in memory, generate a reduction tree
                # which successively sums groups of intermediate results until we get a
                # final result.
                N_per_sum = 3
                while len(chunk_futures) > 1:
                    new_futures = []
                    for i in range(0, len(chunk_futures), N_per_sum):
                        tmp = chunk_futures[i : i + N_per_sum]
                        if len(tmp) == 1:
                            new_futures.append(tmp[0])
                        else:
                            new_futures.append(client.submit(np.sum, tmp, axis=0))
                    chunk_futures = new_futures

                # The final sum is the only future left. Take the inverse FFT.
                ping_result = client.submit(np.fft.ifft, chunk_futures[0])
                del chunk_futures

                # Wait until the result is available and store.
                distributed.wait(ping_result)
                storage["results"][p, r, :] = ping_result.result()
                del ping_result

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

from distributed.lock import Lock
import numpy as np
from numpy.typing import ArrayLike
import pytest
import zarr

from openstb.simulator.controller import simple_points
from openstb.simulator.distortion.environmental import GeometricSpreading
from openstb.simulator.environment.invariant import InvariantEnvironment
from openstb.simulator.plugin import abc
from openstb.simulator.system import GenericSystem
from openstb.simulator.system.ping_times import ConstantInterval
from openstb.simulator.system.signal import LFMChirp
from openstb.simulator.system.trajectory import Linear
from openstb.simulator.system.transducer import GenericTransducer
from openstb.simulator.target.points import SinglePoint
from openstb.simulator.travel_time.stop_and_hop import StopAndHop


@pytest.mark.parametrize("amplitude", [1, 800])
@pytest.mark.parametrize("spreading", ["none", "common", "rx"])
def test_controller_simple_points_chunk_amplitude(amplitude: float, spreading: str):
    """controller.simple_points: amplitude of single chunk"""
    # Generate a simple 1kHz (relative to baseband) sine wave.
    fs = 40e3
    T = 10e-3
    t = np.arange(0, T * 10, 1 / fs)
    f_signal = 91e3
    s = amplitude * np.exp(2j * np.pi * 1e3 * (t - 250e-6))
    s[t >= T] = 0

    # Convert to frequency domain and check amplitude is what we expect (remembering
    # that the signal only fills 10% of the trace).
    f_bb = f_signal - 1e3
    f = np.fft.fftshift(np.fft.fftfreq(len(t), 1 / fs) + f_bb)
    S = np.fft.fftshift(np.fft.fft(s, norm="forward"))
    assert np.allclose(np.abs(S).max(), amplitude / 10)

    # Create common setting structure.
    common = simple_points.CommonSettings(
        f=f,
        S=S,
        signal_frequency_bounds=(f_signal - 1e3, f_signal + 1e3),
        baseband_frequency=f_signal - 1e3,
        travel_time=StopAndHop(),
        trajectory=Linear([0, 0, 0], [0, 1, 0], 1.5),
        environment=InvariantEnvironment(35, 1500, 10),
        tx_position=np.array([0, 0, 0]),
        tx_ori=np.array([1, 0, 0, 0]),
        distortion=[GeometricSpreading(power=1)] if spreading == "common" else [],
    )

    # Simulate two targets spaced by more than signal length.
    E = simple_points._point_simulation_chunk(
        position=np.array([[25, 0, 0], [50, 0, 0]]),
        reflectivity=np.array([1, 1]),
        ping_time=0,
        rx_position=np.array([0, 0, 0]),
        rx_ori=np.array([1, 0, 0, 0]),
        rx_distortion=[GeometricSpreading(power=1)] if spreading == "rx" else [],
        common=common,
        max_t=T * 10,
    )

    # Transform back to the time domain and check each target.
    e = np.fft.ifft(np.fft.ifftshift(E), norm="forward")
    for r in (25, 50):
        # Crop out the centre of the response and check the amplitude.
        t0 = 2 * r / 1500
        idx = np.where((t >= t0) & (t <= (t0 + T)))[0][10:-10]
        expected = amplitude
        if spreading != "none":
            expected /= r**2
        assert np.allclose(np.mean(np.abs(e[idx])), expected)

    # Check areas before first target and after second are close to zero.
    idx = np.where(t < (50 / 1500))[0][:-50]
    assert np.allclose(np.abs(e[idx]) / amplitude, 0, rtol=0, atol=3e-3)
    idx = np.where(t > ((100 / 1500) + T))[0][50:]
    assert np.allclose(np.abs(e[idx]) / amplitude, 0, rtol=0, atol=3e-3)


@pytest.mark.parametrize("spl", [120, 175.5])
@pytest.mark.parametrize("spreading", [False, True])
def test_controller_simple_points_amplitude(
    test_cluster: abc.DaskCluster, tmp_path: Path, spl: float, spreading: bool, subtests
):
    """controller.simple_points: amplitude of individual points"""

    # Define the configuration of the simulation.
    config: simple_points.SimplePointConfig = {
        "dask_cluster": test_cluster,
        "system": GenericSystem(
            transmitter=GenericTransducer(position=[0, 0, 0], orientation=[1, 0, 0, 0]),
            receivers=[
                GenericTransducer(position=[0, 0, 0], orientation=[1, 0, 0, 0]),
                GenericTransducer(position=[0.2, 0, 0], orientation=[1, 0, 0, 0]),
            ],
            signal=LFMChirp(110e3, 130e3, 10e-3, spl),
        ),
        "trajectory": Linear([0, 0, 0], [5, 0, 0], 1.0),
        "ping_times": ConstantInterval(1, 0, 0.2),
        "targets": [SinglePoint([0, 50, 0], 1), SinglePoint([0, 25, 0], 1)],
        "environment": InvariantEnvironment(35, 1480, 10),
        "travel_time": StopAndHop(),
        "distortion": [GeometricSpreading(power=1)] if spreading else [],
    }

    # Create the controller.
    fs = 30e3
    max_t = 100e-3
    Ns = int(np.round(max_t * fs))
    controller = simple_points.SimplePointSimulation(
        result_filename=tmp_path / "results.zarr",
        points_per_chunk=1,
        sample_rate=fs,
        baseband_frequency=120e3,
        max_samples=Ns,
    )

    # Run the simulation and load the results.
    controller.run(config)
    store = zarr.storage.LocalStore(tmp_path / "results.zarr")
    results = zarr.open_group(store=store, mode="r")

    # Check expected variables and attributes are present.
    assert set(results.keys()) == {"pressure", "ping_start_time", "sample_time"}
    assert results.attrs.keys() == {"baseband_frequency", "sample_rate"}
    assert np.isclose(results.attrs["baseband_frequency"], 120e3)  # type:ignore[arg-type]
    assert np.isclose(results.attrs["sample_rate"], 30e3)  # type:ignore[arg-type]

    # Check the sample times are what we expect.
    sample_time: np.ndarray = results["sample_time"][:]  # type:ignore[assignment,index]
    assert sample_time.ndim == 1
    assert len(sample_time) == Ns
    assert np.allclose(sample_time, np.arange(Ns) / fs)

    # And the ping start times.
    ping_time: np.ndarray = results["ping_start_time"][:]  # type:ignore[assignment,index]
    assert ping_time.ndim == 1
    assert len(ping_time) == 5
    assert np.allclose(ping_time, np.arange(5))

    # Check pressure is there and has the expected shape.
    pressure: np.ndarray = results["pressure"][:]  # type:ignore[assignment,index]
    assert pressure.ndim == 3
    assert pressure.shape == (5, 2, Ns)

    # Position of the tx and rx at ping start (since we're using stop-and-hop).
    tx_pos = np.stack((np.arange(5), np.zeros(5), np.zeros(5)), axis=-1)
    rx_pos = np.zeros((5, 2, 3), dtype=float)
    rx_pos[:, 0, :] = tx_pos
    rx_pos[:, 1, :] = tx_pos + [0.2, 0, 0]

    # Check the pressure echoed from each target.
    for x in ([0, 50, 0], [0, 25, 0]):
        # Two-way range to target and corresponding travel time.
        r_tx = np.sqrt(((tx_pos - x) ** 2).sum(axis=-1))
        r_rx = np.sqrt(((rx_pos - x) ** 2).sum(axis=-1))
        twtt = (r_tx[:, None] + r_rx) / 1480.0

        # Expected pressure.
        pa = 1e-6 * 10 ** (spl / 20)
        if spreading:
            pa = pa / r_tx
            pa = pa[:, None] / r_rx

        # Index of first sample in target response. argmax() returns the first
        # occurrence of the maximum if there are multiple.
        first_idx = np.argmax(sample_time[:, None, None] >= twtt, axis=0)

        # Extract the corresponding chunk of the result.
        N_sig = int(np.floor(10e-3 * fs))
        tgt_pressure = np.empty((5, 2, N_sig), dtype=complex)
        for p in range(5):
            for c in range(2):
                idx = first_idx[p, c]
                tgt_pressure[p, c] = pressure[p, c, idx : idx + N_sig]

        # Compute the mean. As we used a signal defined as a complex exponential, this
        # is equivalent to the RMS.
        actual = np.mean(np.abs(tgt_pressure), axis=-1)
        with subtests.test(msg=f"target position {x}"):
            assert np.allclose(actual, pa, atol=0, rtol=0.001)


@pytest.mark.parametrize("spl", [120, 201.3])
@pytest.mark.parametrize("spreading", [False, True])
def test_controller_simple_points_distortion_amplitude(
    test_cluster: abc.DaskCluster, tmp_path: Path, spl: float, spreading: bool
):
    """controller.simple_points: amplitude passed to Distortion is correct"""
    # Sampling settings.
    fs = 30e3
    max_t = 100e-3
    Ns = int(np.round(max_t * fs))
    duration = 10e-3

    # Create our test plugin. We cannot do this at the top-level of the module as it
    # cannot be serialized (Dask would include the path in the serialization, and the
    # tests module is not available to the workers).
    class CheckDistortionAmplitude(abc.Distortion):
        def __init__(self, base_path: Path, fs: float, duration: float):
            self.lock = Lock()
            self.fn = base_path / "rms.txt"
            self.fs = fs
            self.duration = duration

        def apply(
            self,
            ping_time: float,
            f: ArrayLike,
            S: ArrayLike,
            baseband_frequency: float,
            environment: abc.Environment,
            signal_frequency_bounds: tuple[float, float],
            tt_result: abc.TravelTimeResult,
        ) -> np.ndarray:
            # Find the index the echo should start at.
            assert tt_result.travel_time.shape == (1, 1)
            t0 = float(tt_result.travel_time.squeeze())
            idx = int(np.ceil(t0 * self.fs))

            # Take the inverse FFT and extract the echo. We skip the first and last
            # samples as the echo may not be aligned to the sampling.
            S = np.array(S)
            s = np.fft.ifft(np.fft.ifftshift(S.squeeze()), norm="forward")
            N = int(np.floor(self.duration * self.fs))
            echo = s[idx + 1 : idx + N - 2]

            # Store the current RMS of the echo along with the TX and RX ranges.
            rms = float(np.mean(np.abs(echo)))
            tx = float(tt_result.tx_path_length.squeeze())
            rx = float(tt_result.tx_path_length.squeeze())
            self.lock.acquire()
            with self.fn.open("a") as results:
                results.write(f"{ping_time} {tx} {rx} {rms}\n")
            self.lock.release()

            # We don't actually modify the signal.
            return S

    # Create an instance and add to a receiver.
    checker = CheckDistortionAmplitude(tmp_path, fs, duration)
    receivers = [
        GenericTransducer(position=[0, 0, 0], orientation=[1, 0, 0, 0]),
    ]
    receivers[0]._distortion.append(checker)

    # Define the configuration of the simulation.
    config: simple_points.SimplePointConfig = {
        "dask_cluster": test_cluster,
        "system": GenericSystem(
            transmitter=GenericTransducer(position=[0, 0, 0], orientation=[1, 0, 0, 0]),
            receivers=receivers,
            signal=LFMChirp(110e3, 130e3, duration, spl),
        ),
        "trajectory": Linear([0, 0, 0], [5, 0, 0], 1.0),
        "ping_times": ConstantInterval(1, 0, 0.2),
        "targets": [SinglePoint([0, 50, 0], 0.6)],
        "environment": InvariantEnvironment(35, 1480, 10),
        "travel_time": StopAndHop(),
        "distortion": [GeometricSpreading(1)] if spreading else [],
    }

    # Create the controller.
    controller = simple_points.SimplePointSimulation(
        result_filename=tmp_path / "results.zarr",
        points_per_chunk=1,
        sample_rate=fs,
        baseband_frequency=120e3,
        max_samples=Ns,
    )

    # Run the simulation and load the amplitudes recorded by the plugin.
    controller.run(config)
    rms = np.loadtxt(tmp_path / "rms.txt")

    # Expected pressure. Note that the target has a reflectivity of 0.6.
    pa = np.array(0.6 * 1e-6 * 10 ** (spl / 20))
    if spreading:
        pa = pa / rms[:, 1]
        pa = pa / rms[:, 2]

    assert np.allclose(rms[:, 3], pa, atol=0, rtol=0.001)

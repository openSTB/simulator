# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import pytest
from scipy import interpolate

from openstb.simulator.plugin.abc import SignalWindow
from openstb.simulator.system import signal


# For when we want to trigger window behaviour without changing the signal.
class UniformWindow(SignalWindow):
    def get_samples(
        self, t: ArrayLike, duration: float, fill_value: float = 0
    ) -> np.ndarray:
        t = np.asarray(t)
        samples = np.full_like(t, 1.0)
        samples[t < 0] = 0
        samples[t > duration] = 0
        return samples


@pytest.mark.parametrize(
    "f_start,f_stop,duration,spl,rms_after_window",
    [
        (70e3, 90e3, 10e-3, 120, False),
        (90e3, 70e3, 12.5e-3, 180, False),
        (300e3, 330e3, 9e-3, 180, True),
    ],
)
def test_system_signal_lfm(f_start, f_stop, duration, spl, rms_after_window):
    """system.signal: LFM signal without windowing"""
    sig = signal.LFMChirp(
        f_start,
        f_stop,
        duration,
        spl,
        rms_after_window=True,
        window=UniformWindow() if rms_after_window else None,
    )
    assert sig.duration == pytest.approx(duration)
    assert sig.minimum_frequency == pytest.approx(min(f_start, f_stop))
    assert sig.maximum_frequency == pytest.approx(max(f_start, f_stop))

    # This should be disabled if no window was set, even though we set True in init.
    assert sig.rms_after_window == rms_after_window

    # Sample at twice Nyquist.
    BW = np.abs(f_start - f_stop)
    dt = 0.5 / BW

    # Magnitude of the complex exponential is equivalent to RMS.
    A = 1e-6 * 10 ** (spl / 20)

    # Include samples either side and check the magnitude. Skip the edge samples in this
    # as they depend on relative timing of samples.
    t = np.arange(-400 * dt, duration + 400 * dt, dt)
    baseband_freq = (f_start + f_stop) / 2
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s[:399]), 0)
    assert np.allclose(np.abs(s[-399:]), 0)
    assert np.allclose(np.abs(s[401:-401]), A)

    # Check the instantaneous frequency is what we expect. We ignore the first and last
    # sample of the chirp due to edge effects in the calculation.
    inst_f = np.gradient(np.unwrap(np.angle(s)), dt) / (2 * np.pi)
    if f_start < f_stop:
        expected = -(BW / 2) + (t / duration) * BW
    else:
        expected = (BW / 2) - (t / duration) * BW
    assert np.allclose(inst_f[401:-401], expected[401:-401])

    # Repeat with a non-centred baseband frequency.
    baseband_freq += BW / 3
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s[:399]), 0)
    assert np.allclose(np.abs(s[-399:]), 0)
    assert np.allclose(np.abs(s[401:-401]), A)
    inst_f = np.gradient(np.unwrap(np.angle(s)), dt) / (2 * np.pi)
    assert np.allclose(inst_f[401:-401], expected[401:-401] - BW / 3)


def test_system_signal_lfm_window():
    """system.signal: LFM signal windowing"""
    sig = signal.LFMChirp(
        110e3, 90e-3, 0.01, rms_spl=120, rms_after_window=False, window={"name": "hann"}
    )
    t = np.linspace(0, 0.01, 151)
    baseband_freq = 100e3
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s), np.cos(np.pi * (t - 0.005) / 0.01) ** 2)


@pytest.mark.parametrize(
    "spl,window",
    [
        (180, None),
        (200, {"name": "hann"}),
        (195.5, {"name": "tukey", "parameters": {"alpha": 0.15}}),
    ],
)
def test_system_signal_lfm_spl(spl, window):
    """system.signal: LFM signal RMS SPL options"""
    sig = signal.LFMChirp(
        70e3,
        100e3,
        0.008,
        rms_spl=spl,
        rms_after_window=window is not None,
        window=window,
    )
    magtol = 0.005 if window else 1e-7

    t = np.arange(0.5e-3, 7.5e-3, 1 / (75e3))
    s = sig.sample(t, 85e3)
    rms = np.sqrt(np.mean(np.abs(s) ** 2))
    expected = 1e-6 * 10 ** (spl / 20)
    assert np.allclose(rms, expected, rtol=magtol)


def test_system_signal_lfm_rms_variable():
    """system.signal: LFM signal with variable sample spacing"""
    sig = signal.LFMChirp(
        200e3,
        220e3,
        10e-3,
        rms_spl=180,
        rms_after_window=True,
        window={"name": "tukey", "parameters": {"alpha": 0.5}},
    )

    # Generate a time vector with jitter. We go well above the Nyquist limit for the
    # chirp for good interpolation later.
    fs = 180e3
    N = int(np.round(2 * 10e-3 * fs))
    rng = np.random.default_rng(88174571)
    dt = rng.uniform(0.2 / fs, 1.2 / fs, N)
    t = np.cumsum(dt)
    t = t[t <= 10e-3]

    # We will shuffle the time vector to ensure that it is not assumed to be sorted.
    idx = rng.permutation(np.arange(len(t)))
    revidx = np.argsort(idx)

    # Sample the signal and sort back into order.
    s = sig.sample(t[idx], 210e3)[revidx]

    # Interpolate to a regular spacing and check.
    reg = interpolate.CubicSpline(t, s)
    t_reg = np.arange(0, 10e-3, 4 / fs)
    s_reg = reg(t_reg)
    rms_reg = np.sqrt(np.mean(np.abs(s_reg) ** 2))
    expected = 1e-6 * 10 ** (180 / 20)
    assert np.isclose(rms_reg, expected, atol=0.2, rtol=0)


def test_system_signal_lfm_error():
    """system.signal: LFM signal error handling"""
    with pytest.raises(ValueError, match="no .+window plugin .+ installed"):
        signal.LFMChirp(100e3, 140e3, 0.08, 120, window={"name": "a_window_name"})

    with pytest.raises(TypeError, match="missing .+ argument: 'alpha'"):
        signal.LFMChirp(100e3, 140e3, 0.08, 120, window={"name": "tukey"})

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.system import signal


@pytest.mark.parametrize(
    "f_start,f_stop,duration,rms_after_window",
    [
        (70e3, 90e3, 10e-3, False),
        (90e3, 70e3, 12.5e-3, False),
        (300e3, 330e3, 9e-3, True),
    ],
)
def test_system_signal_lfm(f_start, f_stop, duration, rms_after_window):
    """system.signal: LFM signal without windowing"""
    sig = signal.LFMChirp(
        f_start, f_stop, duration, 120, rms_after_window=rms_after_window
    )
    assert sig.duration == pytest.approx(duration)
    assert sig.minimum_frequency == pytest.approx(min(f_start, f_stop))
    assert sig.maximum_frequency == pytest.approx(max(f_start, f_stop))

    # Sample at twice Nyquist.
    BW = np.abs(f_start - f_stop)
    dt = 0.5 / BW

    # Include one sample either side and check the magnitude. We expand the tolerance if
    # we apply the RMS scaling after windowing as this calculates the current RMS
    # instead of using the theoretical value.
    magtol = 0.005 if rms_after_window else 1e-8
    t = np.arange(-dt, duration + dt, dt)
    baseband_freq = (f_start + f_stop) / 2
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s[0]), 0)
    assert np.allclose(np.abs(s[-1]), 0)
    assert np.allclose(np.abs(s[1:-1]), 1, rtol=magtol)

    # Check the instantaneous frequency is what we expect. We ignore the first and last
    # sample of the chirp due to edge effects in the calculation.
    inst_f = np.gradient(np.unwrap(np.angle(s)), dt) / (2 * np.pi)
    if f_start < f_stop:
        expected = -(BW / 2) + (t / duration) * BW
    else:
        expected = (BW / 2) - (t / duration) * BW
    assert np.allclose(inst_f[2:-2], expected[2:-2])

    # Repeat with a non-centred baseband frequency.
    baseband_freq += BW / 3
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s[0]), 0)
    assert np.allclose(np.abs(s[-1]), 0)
    assert np.allclose(np.abs(s[1:-1]), 1, rtol=magtol)
    inst_f = np.gradient(np.unwrap(np.angle(s)), dt) / (2 * np.pi)
    assert np.allclose(inst_f[2:-2], expected[2:-2] - BW / 3)


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


def test_system_signal_lfm_error():
    """system.signal: LFM signal error handling"""
    with pytest.raises(ValueError, match="no .+window plugin .+ installed"):
        signal.LFMChirp(100e3, 140e3, 0.08, 120, window={"name": "a_window_name"})

    with pytest.raises(TypeError, match="missing .+ argument: 'alpha'"):
        signal.LFMChirp(100e3, 140e3, 0.08, 120, window={"name": "tukey"})

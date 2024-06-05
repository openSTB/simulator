# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.system import signal


@pytest.mark.parametrize(
    "f_start,f_stop,duration",
    [(70e3, 90e3, 10e-3), (90e3, 70e3, 12.5e-3), (300e3, 330e3, 9e-3)],
)
def test_system_signal_lfm(f_start, f_stop, duration):
    """system.signal: LFM signal without windowing"""
    sig = signal.LFMChirp(f_start, f_stop, duration)
    assert sig.duration == pytest.approx(duration)

    # Sample at twice Nyquist.
    BW = np.abs(f_start - f_stop)
    dt = 0.5 / BW

    # Include one sample either side and check the magnitude.
    t = np.arange(-dt, duration + dt, dt)
    baseband_freq = (f_start + f_stop) / 2
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s[0]), 0)
    assert np.allclose(np.abs(s[-1]), 0)
    assert np.allclose(np.abs(s[1:-1]), 1)

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
    assert np.allclose(np.abs(s[1:-1]), 1)
    inst_f = np.gradient(np.unwrap(np.angle(s)), dt) / (2 * np.pi)
    assert np.allclose(inst_f[2:-2], expected[2:-2] - BW / 3)


def test_system_signal_lfm_window():
    """system.signal: LFM signal windowing"""
    sig = signal.LFMChirp(110e3, 90e-3, 0.01, "hann")
    t = np.linspace(0, 0.01, 151)
    baseband_freq = 100e3
    s = sig.sample(t, baseband_freq)
    assert np.allclose(np.abs(s), np.cos(np.pi * (t - 0.005) / 0.01) ** 2)


def test_system_signal_lfm_error():
    """system.signal: LFM signal error handling"""
    with pytest.raises(ValueError, match="unknown window"):
        signal.LFMChirp(100e3, 140e3, 0.08, "a_window_name")

    with pytest.raises(ValueError, match="Tukey window requires alpha"):
        signal.LFMChirp(100e3, 140e3, 0.08, "tukey")

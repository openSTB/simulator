# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
from scipy.signal import get_window

from openstb.simulator.system import signal_windows


@pytest.mark.parametrize("fill", [0, -1])
@pytest.mark.parametrize(
    "windowcls,params,scipy_spec",
    [
        (signal_windows.BlackmanWindow, {}, "blackman"),
        (signal_windows.BlackmanHarrisWindow, {}, "blackmanharris"),
        (
            signal_windows.GeneralisedCosineWindow,
            {"coefficients": [0.6, 0.4, 0.3]},
            ("general_cosine", [0.6, 0.4, 0.3]),
        ),
        (
            signal_windows.HammingWindow,
            {"mode": "optimal"},
            ("general_hamming", 0.53836),
        ),
        (
            signal_windows.HammingWindow,
            {"mode": "original"},
            ("general_hamming", 25 / 46),
        ),
        (signal_windows.HammingWindow, {"mode": "scipy"}, "hamming"),
        (signal_windows.HannWindow, {}, "hann"),
        (signal_windows.NuttallWindow, {}, "nuttall"),
        (signal_windows.TukeyWindow, {"alpha": 0.2}, ("tukey", 0.2)),
        (signal_windows.TukeyWindow, {"alpha": 0.13}, ("tukey", 0.13)),
    ],
)
def test_system_signal_windows_scipy(windowcls, params, scipy_spec, fill):
    """system.signal_windows: compare window samples to SciPy"""
    # Check the fill value is correctly applied.
    win = windowcls(**params)
    ours = win.get_samples(np.arange(-5, 23), 20, fill)
    assert np.allclose(ours[:5], fill)
    assert np.allclose(ours[-2:], fill)

    # Compare the values in the valid region to SciPy.
    scipy = get_window(scipy_spec, 21, fftbins=False)
    assert np.allclose(ours[5:-2], scipy)


def test_system_signal_windows_gencos_error():
    """system.signal_windows: error handling of generalised cosine window"""
    with pytest.raises(ValueError, match="at least one coefficient needed"):
        signal_windows.GeneralisedCosineWindow([])


def test_system_signal_windows_hamming_error():
    """system.signal_windows: error handling of Hamming window"""
    with pytest.raises(ValueError, match="unknown Hamming mode"):
        signal_windows.HammingWindow(mode="something")


def test_system_signal_windows_tukey_error():
    """system.signal_windows: error handling of Tukey window"""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        signal_windows.TukeyWindow(alpha=1.2)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        signal_windows.TukeyWindow(alpha=-0.3)

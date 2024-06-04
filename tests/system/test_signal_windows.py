# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
from scipy.signal import get_window

from openstb.simulator.system import signal_windows


def test_system_signal_windows_get():
    """system.signal_windows: get_window_func() can return all available windows"""
    for name in dir(signal_windows):
        if name.endswith("_window"):
            ref_name = name[:-7]
            expected = getattr(signal_windows, name)
            assert signal_windows.get_window_func(ref_name) == expected


def test_system_signal_windows_get_error():
    """system.signal_windows: get_window_func() error handling"""
    with pytest.raises(ValueError, match="unknown window missing"):
        signal_windows.get_window_func("missing")


@pytest.mark.parametrize("fill", [0, -1])
@pytest.mark.parametrize(
    "name,params,scipy_spec",
    [
        ("blackman", {}, "blackman"),
        ("blackman_harris", {}, "blackmanharris"),
        (
            "generalised_cosine",
            {"coefficients": [0.6, 0.4, 0.3]},
            ("general_cosine", [0.6, 0.4, 0.3]),
        ),
        ("hamming", {"optimal": True}, ("general_hamming", 0.53836)),
        ("hamming", {"optimal": False}, ("general_hamming", 25 / 46)),
        ("hann", {}, "hann"),
        ("nuttall", {}, "nuttall"),
        ("tukey", {"alpha": 0.2}, ("tukey", 0.2)),
        ("tukey", {"alpha": 0.13}, ("tukey", 0.13)),
    ],
)
def test_system_signal_windows_scipy(name, params, scipy_spec, fill):
    """system.signal_windows: compare window samples to SciPy"""
    # Check the fill value is correctly applied.
    ours = signal_windows.get_window_func(name)(np.arange(-5, 23), 20, fill, **params)
    assert np.allclose(ours[:5], fill)
    assert np.allclose(ours[-2:], fill)

    # Compare the values in the valid region to SciPy.
    scipy = get_window(scipy_spec, 21, fftbins=False)
    assert np.allclose(ours[5:-2], scipy)


@pytest.mark.parametrize("name", ["blackman", "blackman_harris", "hann", "nuttall"])
def test_system_signal_windows_noparam_error(name):
    """system.signal_windows: error handling of windows which take no parameters"""
    func = signal_windows.get_window_func(name)
    with pytest.raises(ValueError, match="does not accept any parameters"):
        func([0, 1, 2, 3, 4, 5], 3, beta=3)


def test_system_signal_windows_gencos_error():
    """system.signal_windows: error handling of generalised cosine window"""
    with pytest.raises(ValueError, match="needs a list of coefficients"):
        signal_windows.generalised_cosine_window([0, 1, 2], 1)
    with pytest.raises(ValueError, match="unexpected parameter given"):
        signal_windows.generalised_cosine_window(
            [0, 1, 2], 1, coefficients=[0.5, 0.3], extra=7
        )
    with pytest.raises(ValueError, match="unexpected parameters given"):
        signal_windows.generalised_cosine_window(
            [0, 1, 2], 1, coefficients=[0.5, 0.3], extra=7, other=2
        )


def test_system_signal_windows_hamming_error():
    """system.signal_windows: error handling of Hamming window"""
    with pytest.raises(ValueError, match="unexpected parameter given"):
        signal_windows.hamming_window([0, 1, 2], 1, extra=7)
    with pytest.raises(ValueError, match="unexpected parameters given"):
        signal_windows.hamming_window([0, 1, 2], 1, extra=7, other=2)


def test_system_signal_windows_tukey_error():
    """system.signal_windows: error handling of Tukey window"""
    with pytest.raises(ValueError, match="Tukey window requires alpha"):
        signal_windows.tukey_window([0, 1, 2], 1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        signal_windows.tukey_window([0, 1, 2], 1, alpha=1.2)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        signal_windows.tukey_window([0, 1, 2], 1, alpha=-0.3)
    with pytest.raises(ValueError, match="unexpected parameter given"):
        signal_windows.tukey_window([0, 1, 2], 1, alpha=0.2, extra=7)
    with pytest.raises(ValueError, match="unexpected parameters given"):
        signal_windows.tukey_window([0, 1, 2], 1, alpha=0.2, extra=7, other=2)

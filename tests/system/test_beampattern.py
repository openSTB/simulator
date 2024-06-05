# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.system import beampattern


def test_system_beampattern_rect_error():
    """system.beampattern: rectangular beampattern error handling"""
    with pytest.raises(ValueError, match="width .+ cannot be negative"):
        beampattern.Rectangular(-0.05, 0.03)
    with pytest.raises(ValueError, match="height .+ cannot be negative"):
        beampattern.Rectangular(0.05, -0.03)
    with pytest.raises(ValueError, match="cannot have width and height both be zero"):
        beampattern.Rectangular(0, 0)


@pytest.mark.parametrize("f", [120e3, [110e3, 120e3, 130e3, 140e3]])
def test_system_beampattern_rect_horizontal(f):
    """system.beampattern: rectangular beampattern, horizontal only"""
    bp = beampattern.Rectangular(0.04, 0)

    c = 1460.0
    wavelength = c / np.array(f)

    # Angles from normal to evaluate at. If there are multiple frequencies, add a
    # dimension so we can broadcast with the frequency array.
    angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    if wavelength.size > 1:
        angles = angles[:, np.newaxis]
    dirvec = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)

    amp = bp.evaluate(wavelength, dirvec)
    assert np.allclose(amp, np.sinc(0.04 * np.sin(angles) / wavelength))


@pytest.mark.parametrize("f", [75e3, [30e3, 50e3, 70e3]])
def test_system_beampattern_rect_vertical(f):
    """system.beampattern: rectangular beampattern, vertical only"""
    bp = beampattern.Rectangular(0, 0.03)

    c = 1485.0
    wavelength = c / np.array(f)

    angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    if wavelength.size > 1:
        angles = angles[:, np.newaxis]
    dirvec = np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)], axis=-1)

    amp = bp.evaluate(wavelength, dirvec)
    assert np.allclose(amp, np.sinc(0.03 * np.sin(angles) / wavelength))


@pytest.mark.parametrize("f", [60e3, [80e3, 120e3, 160e3, 200e3, 240e3]])
def test_system_beampattern_rect_both(f):
    """system.beampattern: rectangular beampattern, both directions"""
    bp = beampattern.Rectangular(0.04, 0.02)

    c = 1500.0
    wavelength = c / np.array(f)

    angles = np.mgrid[-np.pi / 2 : np.pi / 2 : 361j, -np.pi / 6 : np.pi / 6 : 151j]
    if wavelength.size > 1:
        angles = angles[..., np.newaxis]
    dirvec = np.stack(
        [
            np.cos(angles[0]) * np.cos(angles[1]),
            np.sin(angles[0] * np.cos(angles[1])),
            np.sin(angles[1]),
        ],
        axis=-1,
    )

    amp = bp.evaluate(wavelength, dirvec)
    bp_az = np.sinc(0.04 * np.sin(angles[0] * np.cos(angles[1])) / wavelength)
    bp_el = np.sinc(0.02 * np.sin(angles[1]) / wavelength)
    assert np.allclose(amp, bp_az * bp_el)

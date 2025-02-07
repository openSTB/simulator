# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import quaternionic

from openstb.simulator.distortion import beampattern
from openstb.simulator.environment.invariant import InvariantEnvironment
from openstb.simulator.plugin.abc import TravelTimeResult


def test_scalefactor_beampattern_rect_error():
    """scale_factor.beampattern: rectangular beampattern error handling"""
    with pytest.raises(ValueError, match="width .+ must be positive"):
        beampattern.RectangularBeampattern(-0.05, 0.03, True, True, "centre")
    with pytest.raises(ValueError, match="width .+ must be positive"):
        beampattern.RectangularBeampattern(0, 0.03, True, True, "centre")
    with pytest.raises(ValueError, match="height .+ must be positive"):
        beampattern.RectangularBeampattern(0.05, -0.03, True, True, "centre")
    with pytest.raises(ValueError, match="height .+ must be positive"):
        beampattern.RectangularBeampattern(0.05, -0, True, True, "centre")

    with pytest.raises(ValueError, match="at least one of transmit and receive"):
        beampattern.RectangularBeampattern(0.05, 0.03, False, False, "all")

    with pytest.raises(ValueError, match="unknown value for frequency"):
        beampattern.RectangularBeampattern(0.05, 0.03, True, True, "some of them")

    with pytest.raises(ValueError, match="at least one of horizontal and vertical"):
        beampattern.RectangularBeampattern(
            0.05, 0.03, True, True, "all", horizontal=False, vertical=False
        )


@pytest.mark.parametrize(
    "frequency,tx,rx",
    [
        ("min", True, False),
        ("max", False, True),
        ("centre", True, True),
        ("all", True, True),
    ],
)
def test_scalefactor_beampattern_rect_horizontal(frequency, tx, rx):
    """scale_factor.beampattern: rectangular beampattern, horizontal only"""
    bp = beampattern.RectangularBeampattern(
        0.04, 0.03, tx, rx, frequency, vertical=False
    )

    angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    N = len(angles)
    tgt_vector = np.zeros((1, N, 3))
    tgt_vector[0, :, 0] = -np.sin(angles)
    tgt_vector[0, :, 1] = np.cos(angles)

    env = InvariantEnvironment(salinity=35, sound_speed=1500.0, temperature=7.5)
    tt_result = TravelTimeResult(
        travel_time=np.full((1, N), 0.01),
        tx_position=np.zeros(3),
        tx_orientation=quaternionic.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]),
        tx_velocity=np.array([1.5, 0, 0]),
        tx_vector=tgt_vector[0],
        tx_path_length=np.full((1, N), 150.0),
        rx_position=np.zeros((1, N, 3)),
        rx_orientation=quaternionic.array(
            [[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]] * N
        ).reshape(1, N, 4),
        rx_velocity=np.array([[1.5, 0, 0]] * N).reshape(1, N, 3),
        rx_vector=-tgt_vector,
        rx_path_length=np.full((1, N), 150.0),
    )

    amp = bp.apply(
        0, np.array([100e3, 200e3, 300e3]), 1, 0, env, (50e3, 60e3), tt_result
    )

    wavelength = (
        np.array(1500.0)
        / {
            "min": 50e3,
            "max": 60e3,
            "centre": 55e3,
            "all": np.array([[100e3, 200e3, 300e3]]).T,
        }[frequency]
    )
    expected = np.sinc(0.04 * np.sin(angles)[np.newaxis, :] / wavelength)
    if tx and rx:
        expected = expected**2
    assert np.allclose(amp, expected)


@pytest.mark.parametrize(
    "frequency,tx,rx",
    [
        ("min", True, False),
        ("max", False, True),
        ("centre", True, True),
        ("all", True, True),
    ],
)
def test_scalefactor_beampattern_rect_vertical(frequency, tx, rx):
    """scale_factor.beampattern: rectangular beampattern, vertical only"""
    bp = beampattern.RectangularBeampattern(
        0.04, 0.03, tx, rx, frequency, horizontal=False
    )

    angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    N = len(angles)
    tgt_vector = np.zeros((1, N, 3))
    tgt_vector[0, :, 1] = np.cos(angles)
    tgt_vector[0, :, 2] = np.sin(angles)

    env = InvariantEnvironment(salinity=35, sound_speed=1500.0, temperature=7.5)
    tt_result = TravelTimeResult(
        travel_time=np.full((1, N), 0.01),
        tx_position=np.zeros(3),
        tx_orientation=quaternionic.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]),
        tx_velocity=np.array([1.5, 0, 0]),
        tx_vector=tgt_vector[0],
        tx_path_length=np.full((1, N), 150.0),
        rx_position=np.zeros((1, N, 3)),
        rx_orientation=quaternionic.array(
            [[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]] * N
        ).reshape(1, N, 4),
        rx_velocity=np.array([[1.5, 0, 0]] * N).reshape(1, N, 3),
        rx_vector=-tgt_vector,
        rx_path_length=np.full((1, N), 150.0),
    )

    amp = bp.apply(
        0, np.array([100e3, 200e3, 300e3]), 1, 0, env, (50e3, 60e3), tt_result
    )

    wavelength = (
        np.array(1500.0)
        / {
            "min": 50e3,
            "max": 60e3,
            "centre": 55e3,
            "all": np.array([[100e3, 200e3, 300e3]]).T,
        }[frequency]
    )
    expected = np.sinc(0.03 * np.sin(angles)[np.newaxis, :] / wavelength)
    if tx and rx:
        expected = expected**2
    assert np.allclose(amp, expected)


def test_scalefactor_beampattern_rect_both():
    """scale_factor.beampattern: rectangular beampattern, both directions"""
    bp = beampattern.RectangularBeampattern(0.01, 0.02, True, True, "centre")

    angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
    N = len(angles)
    tgt_vector = np.zeros((1, N, 3))
    tgt_vector[0, :, 0] = np.cos(angles) * np.sin(angles)
    tgt_vector[0, :, 1] = np.cos(angles) * np.cos(angles)
    tgt_vector[0, :, 2] = np.sin(angles)

    env = InvariantEnvironment(salinity=35, sound_speed=1500.0, temperature=7.5)
    tt_result = TravelTimeResult(
        travel_time=np.full((1, N), 0.01),
        tx_position=np.zeros(3),
        tx_orientation=quaternionic.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]),
        tx_velocity=np.array([1.5, 0, 0]),
        tx_vector=tgt_vector[0],
        tx_path_length=np.full((1, N), 150.0),
        rx_position=np.zeros((1, N, 3)),
        rx_orientation=quaternionic.array(
            [[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]] * N
        ).reshape(1, N, 4),
        rx_velocity=np.array([[1.5, 0, 0]] * N).reshape(1, N, 3),
        rx_vector=-tgt_vector,
        rx_path_length=np.full((1, N), 150.0),
    )

    amp = bp.apply(
        0, np.array([100e3, 200e3, 300e3]), 1, 0, env, (50e3, 60e3), tt_result
    )

    wavelength = np.array(1500.0 / 55e3)
    expected_az = np.sinc(0.01 * np.sin(angles) * np.cos(angles) / wavelength)
    expected_el = np.sinc(0.02 * np.sin(angles) / wavelength)
    assert np.allclose(amp, (expected_az * expected_el) ** 2)

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import quaternionic

from openstb.simulator.distortion import environmental
from openstb.simulator.environment.invariant import InvariantEnvironment
from openstb.simulator.plugin.abc import TravelTimeResult


@pytest.mark.parametrize("power,N_rx,N_tgt", [(1, 4, 11), (2, 1, 1), (0.5, 1, 25)])
def test_scalefactor_environmental_geospreading(power, N_rx, N_tgt):
    """scale_factor.environmental: geometric spreading loss factor"""
    env = InvariantEnvironment(salinity=35, sound_speed=1500.0, temperature=7.5)

    tgt_vector = np.zeros((N_tgt, 3))
    tgt_vector[:, 1] = 1

    # Create some path lengths for testing.
    r_tx = np.linspace(100, 150, N_tgt)
    r_rx = np.linspace(50, 125, N_rx * N_tgt).reshape(N_rx, N_tgt)

    tt_result = TravelTimeResult(
        travel_time=np.full((1, N_tgt), 0.01),
        tx_position=np.zeros(3),
        tx_orientation=quaternionic.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]),
        tx_velocity=np.array([1.5, 0, 0]),
        tx_vector=tgt_vector[0],
        tx_path_length=r_tx,
        rx_position=np.zeros((1, N_tgt, 3)),
        rx_orientation=quaternionic.array(
            [[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]] * N_tgt
        ).reshape(1, N_tgt, 4),
        rx_velocity=np.array([[1.5, 0, 0]] * N_tgt).reshape(1, N_tgt, 3),
        rx_vector=-tgt_vector,
        rx_path_length=r_rx,
    )

    gs = environmental.GeometricSpreading(power)
    sf = gs.apply(0, 100e3, 1.0, 0, env, (80e3, 120e3), tt_result)

    # Should have shape (N_receiver, N_frequencies, N_targets). Since our factor has no
    # frequency dependency, we set the frequency axis to length 1.
    assert sf.shape == (N_rx, 1, N_tgt)

    # Expand our input path lengths to the appropriate dimensionality and check.
    r_tx = r_tx[np.newaxis, np.newaxis, :]
    r_rx = r_rx[:, np.newaxis, :]
    if power == 1:
        assert np.allclose(sf, 1 / (r_tx * r_rx))
    elif power == 2:
        assert np.allclose(sf, 1 / (r_tx * r_rx) ** 2)
    else:
        assert np.allclose(sf, 1 / np.sqrt(r_tx * r_rx))


@pytest.mark.parametrize("freqmode", ["min", "max", "centre", "all"])
@pytest.mark.parametrize(
    "T,S,pH,z,f,alpha",
    [
        (11, 37, 8.2, 0.1, 108, 39.06447169),
        (4, 8, 7.9, 0.01, 77.2, 7.23171795),
    ],
)
def test_scalefactor_environmental_ansliemccolm(freqmode, T, S, pH, z, f, alpha):
    """scale_factor.environmental: Anslie-McColm attenuation factor"""
    # Convert values from Anslie-McColm paper units to our units.
    z = z * 1000
    f = f * 1000
    atten = environmental.AnslieMcColmAttenuation(freqmode, pH)
    env = InvariantEnvironment(salinity=S, sound_speed=1500.0, temperature=T)

    # Generate some sample path lengths.
    N_rx = 5
    N_tgt = 22
    r_tx = np.linspace(100, 150, N_tgt)
    r_rx = np.linspace(50, 125, N_rx * N_tgt).reshape(N_rx, N_tgt)

    tgt_vector = np.zeros((N_tgt, 3))
    tgt_vector[:, 1] = 1
    tt_result = TravelTimeResult(
        travel_time=np.full((1, N_tgt), 0.01),
        tx_position=np.array([0, 0, z]),
        tx_orientation=quaternionic.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]),
        tx_velocity=np.array([1.5, 0, 0]),
        tx_vector=tgt_vector[0],
        tx_path_length=r_tx,
        rx_position=np.zeros((N_rx, N_tgt, 3)),
        rx_orientation=quaternionic.array(
            [[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]] * N_tgt
        ).reshape(1, N_tgt, 4),
        rx_velocity=np.array([[1.5, 0, 0]] * N_tgt).reshape(1, N_tgt, 3),
        rx_vector=-tgt_vector,
        rx_path_length=r_rx,
    )

    # Set the calculation parameters so they use the desired frequency.
    if freqmode == "min":
        bounds = (f, f + 20e3)
        f_sim = np.linspace(*bounds, 255)
    elif freqmode == "max":
        bounds = (f - 20e3, f)
        f_sim = np.linspace(*bounds, 255)
    elif freqmode == "centre":
        bounds = (f - 10e3, f + 10e3)
        f_sim = np.linspace(*bounds, 255)
    else:
        bounds = (f - 10e3, f + 10e3)
        f_sim = np.full(255, f)

    sf = atten.apply(0, f_sim, 1.0, 0, env, bounds, tt_result)

    # Should have shape (N_receiver, N_frequencies, N_targets). In the single-frequency
    # modes, we set the frequency axis to length 1.
    if freqmode == "all":
        assert sf.shape == (N_rx, 255, N_tgt)
    else:
        assert sf.shape == (N_rx, 1, N_tgt)

    # Expand our input path lengths to the appropriate dimensionality and check.
    r_tx = r_tx[np.newaxis, np.newaxis, :]
    r_rx = r_rx[:, np.newaxis, :]
    atten_db = (r_tx + r_rx) * alpha / 1000
    atten_lin = 10 ** (-atten_db / 20)
    assert np.allclose(sf, atten_lin)


def test_scalefactor_environmental_ansliemccolm_error():
    """scale_factor.environmental: Anslie-McColm attenuation factor error handling"""
    with pytest.raises(ValueError, match="unknown value for frequency"):
        environmental.AnslieMcColmAttenuation("something", 8.1)

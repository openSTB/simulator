# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.environment.invariant import InvariantEnvironment
from openstb.simulator.scattering import lambertian


@pytest.mark.parametrize("scale", [1, 0.25])
@pytest.mark.parametrize("per_target", [False, True])
@pytest.mark.parametrize("tilt", [0, 10])
def test_scattering_lambertian_apply(scale: float, per_target: bool, tilt: float):
    """scattering.lambertian: check LambertianScattering.apply()"""
    plugin = lambertian.LambertianScattering(scale_factor=scale)

    # Simple incident signal.
    t = np.arange(0, 0.01, 1 / 10e3)
    s = np.sin(2 * np.pi * 1e3 * t)

    # Evaluate at different angles.
    angle = np.radians(np.arange(20, 89, 0.5))
    x = np.zeros_like(angle)
    y = np.cos(angle)
    z = np.sin(angle)
    vec_i = np.stack((x, y, z), axis=-1)
    vec_s = np.stack((x, -y, -z), axis=-1)

    tilt = np.radians(tilt)
    normal = [0, np.sin(tilt), -np.cos(tilt)]

    # If desired, modify each incident signal (e.g., a per-target distortion has already
    # been applied).
    if per_target:
        s = np.repeat(s, len(angle)).reshape(-1, len(angle))
        s *= angle

    # Otherwise insert the necessary length-1 axis to meet the expected dimensionality.
    else:
        s = s.reshape(-1, 1)

    S = np.fft.fft(s, axis=0)
    f = np.fft.fftfreq(len(t), 1 / 10e3)

    env = InvariantEnvironment(sound_speed=1500, salinity=35, temperature=10)
    S_scat = plugin.apply(
        f, S, 0, [0, 0, 0], normal, vec_i, vec_s, env, 1e3, (900, 1100)
    )

    # Output dimensions should be expanded if needed.
    assert S_scat.shape == (len(t), len(angle))

    # Intensity should change by the given scale factor and the cosine of the angle.
    # Note that our input uses the grazing angle not the normal angle, so we have to
    # convert that to check.
    s_scat = np.fft.ifft(S_scat, axis=0)
    I_inc = np.sum(np.abs(s) ** 2, axis=0)
    I_scat = np.sum(np.abs(s_scat) ** 2, axis=0)
    assert np.allclose(I_inc * scale * np.cos(np.pi / 2 - angle + tilt), I_scat)

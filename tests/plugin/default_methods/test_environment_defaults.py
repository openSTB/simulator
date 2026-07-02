# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import gsw  # type:ignore[import-untyped]
import numpy as np
import pytest

from openstb.simulator.plugin.abc import Environment

# Try to find the GSW test data. The binary wheels they provide include it.
gsw_test_base = Path(gsw.__file__).parent / "tests"
gsw_test_data = None
for p in gsw_test_base.glob("gsw_cv_v*.npz"):
    gsw_test_data = p


class TestEnvironment(Environment):
    def __init__(self):
        self.data = np.load(gsw_test_data)

    def salinity(self, t, position):
        return self.data["SA_chck_cast"]

    def sound_speed(self, t, position):
        return NotImplementedError("not needed")

    def temperature(self, t, position):
        return self.data["t_chck_cast"]


@pytest.mark.skipif(gsw_test_data is None, reason="GSW test data not available")
def test_plugin_environment_default_density():
    """plugin.abc.Environment: check default density method"""
    env = TestEnvironment()

    # Generate the equivalent test position with the inverse of the transform the
    # density() method will use.
    z = env.data["p_chck_cast"] / 1.00553
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    pos = np.stack((x, y, z), axis=-1)

    rho = env.density(0, pos)
    assert np.allclose(rho, env.data["rho_t_exact"], equal_nan=True)

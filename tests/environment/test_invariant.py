# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.environment.invariant import InvariantEnvironment


def test_environment_invariant():
    """environment: InvariantEnvironment behaviour"""
    values = {
        "salinity": 34.5,
        "sound_speed": 1488.5,
        "temperature": 11.7,
    }
    env = InvariantEnvironment(**values)

    for attr, expected in values.items():
        method = getattr(env, attr)

        val = method(0, [0, 0, 0])
        assert val.shape == ()
        assert np.allclose(val, expected)

        val = method([0, 60, 120], [0, 0, 0])
        assert val.shape == (1,)
        assert np.allclose(val, expected)

        val = method([0, 60, 120], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
        assert val.shape == (1,)
        assert np.allclose(val, expected)

        t = np.array([0, 60, 120])
        val = method(t[:, np.newaxis], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
        assert val.shape == (1, 1)
        assert np.allclose(val, expected)


def test_environment_invariant_error():
    """environment: InvariantEnvironment error handling"""
    env = InvariantEnvironment(salinity=35, sound_speed=1490, temperature=8)

    with pytest.raises(ValueError, match="cannot be broadcast"):
        env.salinity([0, 60], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
    with pytest.raises(ValueError, match="cannot be broadcast"):
        env.sound_speed([0, 60], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
    with pytest.raises(ValueError, match="cannot be broadcast"):
        env.temperature([0, 60], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])

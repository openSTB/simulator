# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator import environment


def test_environment_invariant():
    """environment: InvariantEnvironment behaviour"""
    env = environment.InvariantEnvironment(sound_speed=1488.5)

    c = env.sound_speed(0, [0, 0, 0])
    assert c.shape == ()
    assert np.allclose(c, 1488.5)

    c = env.sound_speed([0, 60, 120], [0, 0, 0])
    assert c.shape == (1,)
    assert np.allclose(c, 1488.5)

    c = env.sound_speed([0, 60, 120], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
    assert c.shape == (1,)
    assert np.allclose(c, 1488.5)

    t = np.array([0, 60, 120])
    c = env.sound_speed(t[:, np.newaxis], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])
    assert c.shape == (1, 1)
    assert np.allclose(c, 1488.5)


def test_environment_invariant_error():
    """environment: InvariantEnvironment error handling"""
    env = environment.InvariantEnvironment(sound_speed=1488.5)
    with pytest.raises(ValueError, match="cannot be broadcast"):
        env.sound_speed([0, 60], [[0, 0, 0], [90, 0, 0], [180, 0, 0]])

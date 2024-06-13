# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import quaternionic

from openstb.simulator import util


@pytest.mark.parametrize(
    "q,v,vprime",
    [
        ([1, 0, 0, 0], [11, 8, 3], [11, 8, 3]),
        ([0, 0, 0, 1], [-3, 4, 5], [3, -4, 5]),
        ([0, 0, 1, 0], [-3, 4, 5], [3, 4, -5]),
        ([0, 1, 0, 0], [-3, 4, 5], [-3, -4, -5]),
        ([np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0], [1, 0, 0], [0, 0, -1]),
    ],
)
def test_util_rotate_elementwise_single(q, v, vprime):
    """util.rotate_elementwise: single quaternion and vector tests"""
    q = quaternionic.array(q)
    assert np.allclose(util.rotate_elementwise(q, v), vprime)


@pytest.mark.parametrize(
    "q_shape,v_shape,shape",
    [
        [(4,), (5, 3), (5, 3)],
        [(1, 4), (5, 3), (5, 3)],
        [(5, 4), (5, 3), (5, 3)],
        [(15, 4), (3,), (15, 3)],
        [(15, 4), (1, 3), (15, 3)],
        [(7, 15, 4), (1, 3), (7, 15, 3)],
        [(7, 15, 4), (7, 15, 3), (7, 15, 3)],
    ],
)
def test_util_rotate_elementwise_broadcast(q_shape, v_shape, shape):
    """util.rotate_elementwise: broadcasting behaves as expected"""

    # Rotations from 90 to 0 around the z axis.
    Nq = int(np.prod(q_shape[:-1]))
    angles = np.linspace(np.pi, 0, Nq)
    q = np.zeros((Nq, 4))
    q[:, 0] = np.cos(angles / 2)
    q[:, 3] = np.sin(angles / 2)
    q = quaternionic.array(q.reshape(q_shape))

    # Vectors of length 1 to N along the x axis.
    Nv = int(np.prod(v_shape[:-1]))
    lengths = np.arange(1, Nv + 1)
    v = np.zeros((Nv, 3))
    v[:, 0] = lengths
    v = v.reshape(v_shape)

    # Expected value.
    Nout = int(np.prod(shape[:-1]))
    expected = np.zeros((Nout, 3))
    expected[:, 0] = lengths * np.cos(angles)
    expected[:, 1] = lengths * np.sin(angles)
    expected = expected.reshape(shape)

    vprime = util.rotate_elementwise(q, v)
    assert vprime.shape == shape
    assert np.allclose(vprime, expected)


def test_util_rotate_elementwise_error():
    """util.rotate_elementwise: errors on non-broadcastable arrays"""
    q = quaternionic.array([[1, 0, 0, 0]] * 5)
    v = np.array([[1, 0, 0]] * 3)
    with pytest.raises(ValueError, match="could not be broadcast"):
        util.rotate_elementwise(q, v)

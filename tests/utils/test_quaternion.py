# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
import quaternionic

from openstb.simulator.utils import quaternion


def q_allclose(q1, q2) -> bool:
    return np.allclose(q1.ndarray, q2) or np.allclose((-q1).ndarray, q2)


def test_utils_quaternion_from_vectors_parallel():
    """utils.quaternion.quaternion_from_vectors: parallel inputs"""
    ref = [1, 0, 0]
    tgt = [1, 0, 0]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (4,)
    assert q_allclose(q, [1, 0, 0, 0])

    tgt = [10, 0, 0]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (4,)
    assert q_allclose(q, [1, 0, 0, 0])

    tgt = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (3, 4)
    assert q_allclose(q, [1, 0, 0, 0])

    ref = [10, 0, 0]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (3, 4)
    assert q_allclose(q, [1, 0, 0, 0])

    ref = [[10, 0, 0], [10, 0, 0], [10, 0, 0]]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (3, 4)
    assert q_allclose(q, [1, 0, 0, 0])


def test_utils_quaternion_from_vectors_antiparallel():
    """utils.quaternion.quaternion_from_vectors: anti-parallel inputs"""
    ref = [-1, 0, 0]
    tgt = [1, 0, 0]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (4,)
    assert q_allclose(q, [0, 0, 0, 1])


def test_utils_quaternion_from_vectors_axes():
    """utils.quaternion.quaternion_from_vectors: with cardinal axes"""
    ref = [1, 0, 0]
    tgt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert np.allclose(q.rotate(ref), tgt)


def test_utils_quaternion_from_vectors_random_tgt():
    """utils.quaternion.quaternion_from_vectors: with random targets"""
    ref = [1, 0, 0]
    rng = np.random.default_rng(56789171761751)
    tgt = rng.uniform(-5, 5, (10, 20, 3))
    q = quaternion.quaternion_from_vectors(ref, tgt)
    assert q.shape == (10, 20, 4)
    assert np.allclose(q.norm, 1)

    # Rotate the reference by this and check it is parallel to the original.
    unit_tgt = q.rotate(ref)
    norm = np.linalg.norm(tgt, axis=-1, keepdims=True)
    assert np.allclose(np.vecdot(unit_tgt, tgt / norm, axis=-1), 1)


def test_utils_quaternion_from_vectors_random_vec():
    """utils.quaternion.quaternion_from_vectors: with random vectors"""
    rng = np.random.default_rng(568264810875)
    ref = rng.uniform(-5, 5, (10, 20, 3))
    tgt = rng.uniform(-5, 5, (10, 20, 3))
    q = quaternion.quaternion_from_vectors(ref, tgt)

    assert q.shape == (10, 20, 4)
    assert np.allclose(q.norm, 1)
    norm_ref = np.linalg.norm(ref, axis=-1, keepdims=True)
    norm_tgt = np.linalg.norm(tgt, axis=-1, keepdims=True)
    gen = quaternion.rotate_elementwise(q, ref)
    assert np.allclose(np.vecdot(tgt / norm_tgt, gen / norm_ref, axis=-1), 1)


def test_utils_quaternion_from_vectors_error():
    """utils.quaternion.quaternion_from_vectors: error reporting"""
    with pytest.raises(ValueError, match="reference.+size 3"):
        quaternion.quaternion_from_vectors([1, 0], [1, 0, 0])
    with pytest.raises(ValueError, match="target.+size 3"):
        quaternion.quaternion_from_vectors([1, 0, 0], [1, 0])

    with pytest.raises(ValueError, match="reference.+single.+or.+same size"):
        quaternion.quaternion_from_vectors([[1, 0, 0], [0, 1, 0]], [1, 0, 0])
    with pytest.raises(ValueError, match="reference.+single.+or.+same size"):
        quaternion.quaternion_from_vectors(
            [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        )


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
def test_utils_rotate_elementwise_single(q, v, vprime):
    """utils.quaternion.rotate_elementwise: single quaternion and vector tests"""
    q = quaternionic.array(q)
    assert np.allclose(quaternion.rotate_elementwise(q, v), vprime)


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
def test_utils_rotate_elementwise_broadcast(q_shape, v_shape, shape):
    """utils.quaternion.rotate_elementwise: broadcasting behaves as expected"""

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

    vprime = quaternion.rotate_elementwise(q, v)
    assert vprime.shape == shape
    assert np.allclose(vprime, expected)


def test_utils_rotate_elementwise_error():
    """utils.quaternion.rotate_elementwise: errors on non-broadcastable arrays"""
    q = quaternionic.array([[1, 0, 0, 0]] * 5)
    v = np.array([[1, 0, 0]] * 3)
    with pytest.raises(ValueError, match="could not be broadcast"):
        quaternion.rotate_elementwise(q, v)

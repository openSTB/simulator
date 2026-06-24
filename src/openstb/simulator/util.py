# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import quaternionic


def quaternion_from_vectors(
    reference: ArrayLike, target: ArrayLike
) -> quaternionic.QArray:
    """Find quaternions to rotate from one vector direction to another.

    Parameters
    ----------
    reference
        The vector defining the direction to rotate from. This must either be shape (3,)
        or the same shape as `target`.
    target
        The vector defining the direction to rotate to. This must have a last axis of
        size 3.

    Returns
    -------
    q
        A quaternion array rotating vectors from the direction of reference to the
        direction of target.

    """
    reference = np.atleast_1d(reference).astype(float)
    if reference.shape[-1] != 3:
        raise ValueError("last axis of reference must have size 3")

    target = np.atleast_1d(target).astype(float)
    if target.shape[-1] != 3:
        raise ValueError("last aixs of target must have size 3")

    if reference.ndim != 1 and reference.shape != target.shape:
        raise ValueError(
            "reference must be a single vector or an array of the same size as target"
        )

    # Normalise the vectors.
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    target /= np.linalg.norm(target, axis=-1, keepdims=True)

    # Calculate the dot product and determine parallel cases.
    dp = np.vecdot(reference, target, axis=-1)  # type:ignore[call-overload]
    parallel = np.isclose(dp, 1)
    antiparallel = np.isclose(dp, -1)

    # Use the cross product to find an axis of rotation. In parallel cases, this will
    # have the length zero. We can choose any axis we want here.
    rotaxis = np.cross(reference, target, axisa=-1, axisb=-1)
    rotaxis[parallel] = [0, 0, 1]
    rotaxis[antiparallel] = [0, 0, 1]
    rotaxis = rotaxis / np.linalg.norm(rotaxis, axis=-1, keepdims=True)

    # Find the angle of rotation and construct.
    angle = np.arccos(dp)
    ori = np.empty(target.shape[:-1] + (4,), dtype=float)
    ori[..., 0] = np.cos(angle / 2)
    ori[..., 1:] = np.sin(angle / 2)[..., np.newaxis] * rotaxis

    return quaternionic.array(ori)


def rotate_elementwise(q: quaternionic.QArray, v: ArrayLike) -> np.ndarray:
    """Element-wise rotation of vectors with quaternions.

    The `quaternionic.array.rotate` method is like an outer product: it rotates each
    vector by each quaternion. That is, for a quaternion array of shape (M, N, 4) and a
    vector array of (P, Q, 3), its output will have a shape (M, N, P, Q, 3).

    By contrast, this function rotates in an element-wise fashion with standard NumPy
    broadcasting rules. This means a quaternion array of shape (M, N, 4) and a vector
    array of (M, N, 3) will result in an output of shape (M, N, 3).

    Parameters
    ----------
    quat : quaternionic.array
        Array of quaternions defining the rotations.
    v : array-like
        Array of vectors to be rotated.

    Returns
    -------
    vprime : numpy.ndarray
        The rotated vectors.

    """
    # The docstring of quaternionic.array.rotate suggests the following formula.
    v = np.array(v)
    inner = q.scalar[..., np.newaxis] * v + np.cross(q.vector, v, axisa=-1, axisb=-1)
    return v + 2 * np.cross(q.vector, inner) / q.mag2[..., np.newaxis]

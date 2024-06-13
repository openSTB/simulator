# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import quaternionic


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

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.abc import Environment


class InvariantEnvironment(Environment):
    """Environment with space- and time-invariant properties."""

    def __init__(self, sound_speed: float):
        """
        Parameters
        ----------
        sound_speed : float
            The speed of propagation in metres per second.

        """
        self._sound_speed = sound_speed

    def sound_speed(self, t: ArrayLike, position: ArrayLike) -> np.ndarray:
        # Find the broadcast shape of the inputs, ignoring the final axis of position.
        bc_shape = np.broadcast_shapes(np.array(t).shape, np.array(position).shape[:-1])

        # Take our constant and turn it into an array of size 1 along all dimensions.
        return np.array(self._sound_speed).reshape([1] * len(bc_shape))

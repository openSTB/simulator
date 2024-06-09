# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Travel time calculation plugins for the openSTB simulator."""

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.abc import Trajectory, TravelTime


class StopAndHop(TravelTime):
    """Travel time calculated using the stop-and-hop approximation.

    Note that this plugin does not calculate any scale factors to be applied. To model
    effects such as attenuation and spreading loss, separate plugins must be included in
    the simulation setup.

    """

    def calculate(
        self,
        trajectory: Trajectory,
        ping_time: float,
        tx_position: ArrayLike,
        rx_positions: ArrayLike,
        target_positions: ArrayLike,
        sound_speed: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, None]:
        # Find the position and orientation of the vehicle at the start of the ping...
        vehicle_pos = trajectory.position(ping_time)
        vehicle_ori = trajectory.orientation(ping_time)

        # ... and from that the tx and rx positions.
        tx_pos = vehicle_pos + vehicle_ori.rotate(np.asarray(tx_position))
        rx_pos = vehicle_pos + vehicle_ori.rotate(np.asarray(rx_positions))

        # Vector from tx to target and target to rx.
        tx_vec = target_positions - tx_pos
        rx_vec = rx_pos[:, np.newaxis, :] - target_positions

        # Corresponding distances.
        tx_pathlen = np.sqrt(np.sum(tx_vec**2, axis=-1))
        rx_pathlen = np.sqrt(np.sum(rx_vec**2, axis=-1))

        # Travel time is an easy calculation.
        tt = (tx_pathlen + rx_pathlen) / sound_speed

        # Normalise the vectors to give the transmit and receive directions.
        tx_vec /= tx_pathlen[:, np.newaxis]
        rx_vec /= rx_pathlen[:, :, np.newaxis]

        return tt, tx_vec, tx_pathlen, rx_vec, rx_pathlen, None

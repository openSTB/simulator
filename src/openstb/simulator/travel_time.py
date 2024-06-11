# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Travel time calculation plugins for the openSTB simulator."""

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.simulator.abc import Trajectory, TravelTime, TravelTimeResult


class StopAndHop(TravelTime):
    """Travel time calculated using the stop-and-hop approximation.

    Note that this plugin does not calculate any scale factors to be applied. To model
    effects such as attenuation and spreading loss, separate plugins must be included in
    the simulation setup.

    """

    def __init__(self, sound_speed: float):
        """
        Parameters
        ----------
        sound_speed : float
            The speed (in metres/second) that sound travels at in the medium.

        """
        self.sound_speed = sound_speed

    def calculate(
        self,
        trajectory: Trajectory,
        ping_time: float,
        tx_position: ArrayLike,
        tx_orientation: ArrayLike | quaternionic.QArray,
        rx_positions: ArrayLike,
        rx_orientations: ArrayLike | quaternionic.QArray,
        target_positions: ArrayLike,
    ) -> TravelTimeResult:
        # Find the position and orientation of the vehicle at the start of the ping...
        vehicle_pos = trajectory.position(ping_time)
        vehicle_ori = trajectory.orientation(ping_time)

        # ... and from that the tx and rx details.
        tx_pos = vehicle_pos + vehicle_ori.rotate(np.asarray(tx_position))
        tx_ori = quaternionic.array(tx_orientation) * vehicle_ori
        rx_pos = vehicle_pos + vehicle_ori.rotate(np.asarray(rx_positions))
        rx_ori = quaternionic.array(rx_orientations) * vehicle_ori

        # Vector from tx to target and target to rx.
        tx_vec = target_positions - tx_pos
        rx_vec = rx_pos[:, np.newaxis, :] - target_positions

        # Corresponding distances.
        tx_pathlen = np.sqrt(np.sum(tx_vec**2, axis=-1))
        rx_pathlen = np.sqrt(np.sum(rx_vec**2, axis=-1))

        # Travel time is an easy calculation.
        tt = (tx_pathlen + rx_pathlen) / self.sound_speed

        # Normalise the vectors to give the transmit and receive directions.
        tx_vec /= tx_pathlen[:, np.newaxis]
        rx_vec /= rx_pathlen[:, :, np.newaxis]

        # And rotate these into the transducer coordinate systems.
        tx_vec = (~tx_ori).rotate(tx_vec)
        rx_vec = (~rx_ori).rotate(rx_vec)

        return TravelTimeResult(
            travel_time=tt,
            tx_position=tx_pos,
            tx_orientation=tx_ori,
            tx_vector=tx_vec,
            tx_path_length=tx_pathlen,
            rx_position=rx_pos[:, np.newaxis, :],
            rx_orientation=rx_ori[:, np.newaxis, :],
            rx_vector=rx_vec,
            rx_path_length=rx_pathlen,
            scale_factor=None,
        )

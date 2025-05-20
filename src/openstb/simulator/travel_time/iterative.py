# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Travel time calculation plugins for the openSTB simulator."""

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import (
    Environment,
    Trajectory,
    TravelTime,
    TravelTimeResult,
)
from openstb.simulator.util import rotate_elementwise

_ = translations.load("openstb.simulator").gettext


class Iterative(TravelTime):
    """Travel time including vehicle motion during the ping.

    This uses an iterative approach to find the travel time. The initial guess is found
    with the assumption that the vehicle motion at the start of the ping is constant
    throughout the ping. The receiver position at the proposed reception time is found.
    From the corresponding path length an updated travel time is found. This continues
    to iterate until successive travel times are within the requested tolerance.

    This uses the sound speed at the position of the vehicle and time of the ping
    transmission throughout the calculation, and assumes that the sound travels in a
    straight line to and from the targets.

    Note that this plugin does not calculate any scale factors to be applied. To model
    effects such as attenuation and spreading loss, separate plugins must be included in
    the simulation setup.

    """

    def __init__(self, max_iterations: int, tolerance: float):
        """
        Parameters
        ----------
        max_iterations : int
            If a solution is not found within this many iterations, an exception will be
            raised.
        tolerance : float
            The tolerance in seconds. If the travel time estimated in successive
            iterations differ by less than this, they are considered successful.

        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def calculate(
        self,
        trajectory: Trajectory,
        ping_time: float,
        environment: Environment,
        tx_position: ArrayLike,
        tx_orientation: ArrayLike | quaternionic.QArray,
        rx_positions: ArrayLike,
        rx_orientations: ArrayLike | quaternionic.QArray,
        target_positions: ArrayLike,
    ) -> TravelTimeResult:
        # Find the position and orientation of the vehicle at the start of the ping.
        vehicle_pos0 = trajectory.position(ping_time)
        vehicle_vel0 = trajectory.velocity(ping_time)
        vehicle_speed0 = np.linalg.norm(vehicle_vel0)
        vehicle_ori0 = trajectory.orientation(ping_time)

        # Sound speed at transmission.
        sound_speed = environment.sound_speed(ping_time, vehicle_pos0)

        # Transmitter details at the start of the ping.
        tx_pos = vehicle_pos0 + vehicle_ori0.rotate(np.asarray(tx_position))
        tx_ori = quaternionic.array(tx_orientation) * vehicle_ori0

        rx_positions = np.asarray(rx_positions)
        rx_orientations = quaternionic.array(rx_orientations)

        # Vector and range between receivers and targets at start of ping.
        target_positions = np.asarray(target_positions)  # (Nt, 3)
        rx_vec0 = target_positions - rx_positions[:, np.newaxis, :]  # (Nr, Nt, 3)
        rx_r0 = np.linalg.norm(rx_vec0, axis=-1)  # (Nr, Nt)

        # Initial estimate of travel time using the linear velocity approximation.
        vrel = (rx_vec0 * vehicle_vel0).sum(axis=-1)  # (Nr, Nt)
        tt = (2 * rx_r0 * sound_speed - 2 * vrel) / (sound_speed**2 - vehicle_speed0**2)

        # Transmit portion.
        tx_vec = target_positions - tx_pos
        tx_pathlen = np.sqrt(np.sum(tx_vec**2, axis=-1))
        tx_vec /= tx_pathlen[:, np.newaxis]
        tx_tt = tx_pathlen / sound_speed  # (Nt,)

        # Loop for the maximum number of iterations; exit if we converge sooner.
        success = False
        for i in range(self.max_iterations):
            # Times the echoes would be received at with current estimate.
            rx_time = ping_time + tt  # (Nr, Nt)

            # Position and orientation of the vehicle at proposed reception times.
            vehicle_pos_rx = trajectory.position(rx_time)  # (Nr, Nt, 3)
            vehicle_ori_rx = trajectory.orientation(rx_time)  # (Nr, Nt, 4)

            # Corresponding position of receivers.
            rx_pos_rx = vehicle_pos_rx + rotate_elementwise(
                vehicle_ori_rx, rx_positions[:, np.newaxis, :]
            )  # (Nr, Nt, 3)

            # Vector and path length for each echo to proposed receiver positions.
            rx_vec = rx_pos_rx - target_positions
            rx_pathlen = np.sqrt(np.sum(rx_vec**2, axis=-1))

            # Total travel time for these receiver positions.
            new_tt = tx_tt + rx_pathlen / sound_speed

            # If it is close to the previous travel time (which was used to find the
            # receiver positions), we have converged to a solution.
            dt = np.abs(new_tt - tt)
            tt = new_tt
            if np.all(dt < self.tolerance):
                success = True
                break

        # Check why we exited the loop.
        if not success:
            raise RuntimeError(_("could not converge to a travel time"))

        # Put it all together.
        return TravelTimeResult(
            travel_time=tt,
            tx_position=tx_pos,
            tx_orientation=tx_ori,
            tx_velocity=vehicle_vel0,
            tx_vector=tx_vec,
            tx_path_length=tx_pathlen,
            rx_position=rx_pos_rx,
            rx_orientation=rx_orientations[:, np.newaxis] * vehicle_ori_rx,
            rx_velocity=trajectory.velocity(ping_time + tt),
            rx_vector=rx_vec / rx_pathlen[..., np.newaxis],
            rx_path_length=rx_pathlen,
            scale_factor=None,
        )

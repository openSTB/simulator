# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult


class DopplerDistortion(Distortion):
    """Doppler distortion due to the movement of the platform.

    This distortion is performed as follows:

    1. Calculate the speed of the transducers in the direction of the target (positive
       towards the target).

    2. Calculate the Doppler scale factor n = (1 + v_rx/c) / (1 - v_tx/c).

    3. Distort the spectrum according to S_d(f) = S(f/n) / sqrt(n).


    """

    def __init__(self, calculate_c_rx: bool = False):
        """
        Parameters
        ----------
        calculate_c_rx : Boolean
            If True, calculate the platform velocity at the reception time and position
            separately for each target. If False, use the platform velocity at the time
            and position the ping transmission started for all calculations.

        """
        self.calculate_c_rx = calculate_c_rx

    def apply(
        self,
        ping_time: float,
        f: ArrayLike,
        S: ArrayLike,
        baseband_frequency: float,
        environment: Environment,
        signal_frequency_bounds: tuple[float, float],
        tt_result: TravelTimeResult,
    ) -> np.ndarray:
        # Get the sound speeds.
        c_tx = environment.sound_speed(ping_time, tt_result.tx_position)
        if self.calculate_c_rx:
            c_rx = environment.sound_speed(
                ping_time + tt_result.travel_time, tt_result.rx_position
            )
        else:
            c_rx = c_tx

        # Relative velocities between the transducers and targets, with positive meaning
        # movement towards the targets.
        tx_relvel = np.sum(tt_result.tx_vector * tt_result.tx_velocity, axis=-1)
        rx_relvel = np.sum(-tt_result.rx_vector * tt_result.rx_velocity, axis=-1)

        # Doppler distortion coefficient.
        eta = (1 + rx_relvel / c_rx) / (1 - tx_relvel / c_tx)

        # Create interpolators. Disabling extrapolation means the interpolator will set
        # frequencies outside the original spectrum to NaN.
        f = np.array(f)
        S = np.array(S)
        interp_r = PchipInterpolator(f, S.real, axis=1, extrapolate=False)
        interp_i = PchipInterpolator(f, S.imag, axis=1, extrapolate=False)

        # Apply the distortion by interpolation; set frequencies outside to zero.
        f_interp = f / eta
        S_distorted = interp_r(f_interp) + 1j * interp_i(f_interp)
        S_distorted[np.isnan(S_distorted)] = 0
        S_distorted /= np.sqrt(eta)

        return S_distorted

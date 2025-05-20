# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Ping time calculator plugins for the openSTB simulator."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import PingTimes, Trajectory

_ = translations.load("openstb.simulator").gettext


class ConstantInterval(PingTimes):
    """Pings at a constant interval."""

    def __init__(self, interval: float, start_delay: float, end_delay: float):
        """
        Parameters
        ----------
        interval : float
            The ping interval in seconds.
        start_delay : float
            Delay (in seconds) between the start of the trajectory and the first ping.
        end_delay : float
            Delay (in seconds) between the last ping and the end of the trajectory.

        """
        if not interval > 0:
            raise ValueError(_("ping interval must be greater than zero"))
        if start_delay < 0:
            raise ValueError(_("delay of first ping cannot be less than zero"))
        if end_delay < 0:
            raise ValueError(_("delay of last ping cannot be less than zero"))

        self.interval = interval
        self.start_delay = start_delay
        self.end_delay = end_delay

    def calculate(self, trajectory: Trajectory) -> np.ndarray:
        return np.arange(
            self.start_delay, trajectory.duration - self.end_delay, self.interval
        )


class ConstantDistance(PingTimes):
    """Pings at a constant distance.

    The mean speed of the trajectory is used to find the mean interval between pings.
    The position of the system is sampled at a rate greater than this and the cumulative
    distance between these positions is calculated to give an estimated mapping between
    the elapsed time and distance travelled. The ping times are then found by
    interpolating the mapping at the desired positions using a Piecewise Cubic Hermite
    Interpolating Polynomial (PCHIP) interpolator.

    """

    def __init__(
        self,
        distance: float,
        start_offset: float,
        end_offset: float,
        sampling_factor: int = 10,
    ):
        """
        Parameters
        ----------
        distance : float
            The distance between pings in metres.
        start_offset : float
            The distance (in metres) between the start of the trajectory and the
            position of the first ping.
        end_offset : float
            The minimum distance (in metres) between the position of the final ping and
            the end of the trajectory.
        sampling_factor : int
            Sample the system position this many times in each mean interval. For
            example, a mean speed of 2m/s and a distance of 0.5m corresponds to a mean
            interval of 0.25s. A sampling factor of 10 means the trajectory will be
            sampled every 0.025s.

        """
        if not distance > 0:
            raise ValueError(_("ping-to-ping distance must be greater than zero"))
        if start_offset < 0:
            raise ValueError(_("start offset cannot be less than zero"))
        if end_offset < 0:
            raise ValueError(_("end offset cannot be less than zero"))
        if sampling_factor < 1:
            raise ValueError(_("sampling factor cannot be less than one"))

        self.distance = distance
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.sampling_factor = sampling_factor

    def calculate(self, trajectory: Trajectory) -> np.ndarray:
        # Compute the mean interval.
        mean_speed = trajectory.length / trajectory.duration
        mean_interval = self.distance / mean_speed

        # Sample the trajectory position and compute the cumulative distance.
        t = np.arange(0, trajectory.duration, mean_interval / self.sampling_factor)
        pos = trajectory.position(t)
        d = np.zeros_like(t)
        d[1:] = np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=-1))

        # Create a distance to elapsed time interpolator; evaluate at desired distances.
        interp = PchipInterpolator(d, t)
        targets = np.arange(self.start_offset, d[-1] - self.end_offset, self.distance)
        return interp(targets)

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator import abc, plugin


class LFMChirp(abc.Signal):
    """Linear frequency modulated chirp."""

    def __init__(
        self,
        f_start: float,
        f_stop: float,
        duration: float,
        window: plugin.PluginSpec | None = None,
    ):
        """
        Parameters
        ----------
        f_start, f_stop : float
            The start (t=0) and stop (t=duration) frequencies of the chirp.
        duration : float
            The duration of the chirp in seconds.
        window : PluginSpec
            Plugin specification for a signal window to apply to the samples of the
            signal.

        """
        self.f_start = f_start
        self.f_stop = f_stop
        self._duration = duration
        self.window = None if window is None else plugin.signal_window(window)

    @property
    def duration(self) -> float:
        return self._duration

    def sample(self, t: ArrayLike, baseband_frequency: float) -> np.ndarray:
        # Initialise the output array.
        t = np.asarray(t)
        s = np.zeros_like(t, dtype=complex)

        # Find a mask where the times are inside the signal duration.
        valid = (t >= 0) & (t <= self._duration)
        tv = t[valid]

        # Convert the starting frequency to baseband, calculate the chirp rate and from
        # that generate the valid portions of the signal.
        fd = self.f_start - baseband_frequency
        K = (self.f_stop - self.f_start) / self._duration
        s[valid] = np.exp(1j * np.pi * tv * (2 * fd + K * tv))

        if self.window is not None:
            s *= self.window.get_samples(t, self._duration, fill_value=0)

        return s
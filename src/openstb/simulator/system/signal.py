# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin import abc
from openstb.simulator.plugin.loader import signal_window
from openstb.simulator.types import PluginOrSpec


class LFMChirp(abc.Signal):
    """Linear frequency modulated chirp."""

    def __init__(
        self,
        f_start: float,
        f_stop: float,
        duration: float,
        rms_spl: float,
        rms_after_window: bool = True,
        window: PluginOrSpec[abc.SignalWindow] | None = None,
    ):
        """
        Parameters
        ----------
        f_start, f_stop : float
            The start (t=0) and stop (t=duration) frequencies of the chirp.
        duration : float
            The duration of the chirp in seconds.
        rms_spl : float
            The RMS sound pressure level (decibels relative to 1 micropascal) of the
            signal.
        rms_after_window : Boolean
            If True, scale the signal to the desired RMS SPL after applying the window.
            If False, scale before applying the window, meaning the windowed signal will
            have a lower RMS SPL.
        window : PluginSpec
            Plugin specification for a signal window to apply to the samples of the
            signal.

        """
        self.f_start = f_start
        self.f_stop = f_stop
        self._minimum_frequency = min(f_start, f_stop)
        self._maximum_frequency = max(f_start, f_stop)
        self._duration = duration
        self.rms_spl = rms_spl
        self.rms_after_window = rms_after_window
        self.window = None if window is None else signal_window(window)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def minimum_frequency(self) -> float:
        return self._minimum_frequency

    @property
    def maximum_frequency(self) -> float:
        return self._maximum_frequency

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

        # Calculate the source level in Pascal.
        level = 10 ** (self.rms_spl / 20) * 1e-6

        if not self.rms_after_window:
            s *= level

        if self.window is not None:
            s *= self.window.get_samples(t, self._duration, fill_value=0)

        if self.rms_after_window:
            current = np.sqrt(np.mean(np.abs(s) ** 2))
            s *= level / current

        return s

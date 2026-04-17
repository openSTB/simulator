# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin import abc
from openstb.simulator.plugin.loader import signal_window
from openstb.simulator.types import PluginOrSpec


class LFMChirp(abc.Signal):
    r"""Linear frequency modulated chirp.

    For a start frequency $f_a$, stop frequency $f_b$ and chirp length $\tau$, we can
    calculate the chirp rate $K = (f_b - f_a) / \tau$. In the passband, the LFM chirp is
    then

    \[
    s(t) = \exp (j\pi t  (2f_a + Kt))
    \qquad
    0 \leq t \leq \tau,
    \]

    which has a linear instantaneous frequency $f(t) = f_a + Kt$.

    """

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
        f_start
            The start frequency of the chirp.
        f_stop
            The stop frequency of the chirp.
        duration
            The duration of the chirp in seconds.
        rms_spl
            The RMS sound pressure level (decibels relative to 1 micropascal) of the
            signal.
        rms_after_window
            If True, scale the signal to the desired RMS SPL after applying the window.
            If False, scale before applying the window, meaning the windowed signal will
            have a lower RMS SPL. If no window is applied, this parameter is ignored.
        window
            Plugin specification for a signal window to apply to the samples of the
            signal.

        """
        self.f_start = f_start
        self.f_stop = f_stop
        self._minimum_frequency = min(f_start, f_stop)
        self._maximum_frequency = max(f_start, f_stop)
        self._duration = duration
        self.rms_spl = rms_spl
        self.rms_after_window = rms_after_window and (window is not None)
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

        # Calculate the desired source level in Pascal.
        level = 10 ** (self.rms_spl / 20) * 1e-6

        # Not windowing => the current source level is 1Pa.
        if not self.rms_after_window:
            s *= level

        if self.window is not None:
            s *= self.window.get_samples(t, self._duration, fill_value=0)

        if self.rms_after_window:
            # Don't assume the sample times are sorted. argsort() gives us the indices
            # to select to get a sorted array.
            idx = np.argsort(tv)

            # And then sort the valid part of the output.
            t_s = tv[idx]
            s_s = s[valid][idx]

            # Weighting the mean allows for a variable sample spacing.
            dt = np.diff(t_s)
            current = np.sqrt(np.average(np.abs(s_s[1:]) ** 2, weights=dt))

            # Which we can then adjust as desired.
            s *= level / current

        return s


class HFMChirp(abc.Signal):
    r"""Hyperbolic frequency modulated chirp.

    For a start frequency $f_a$, stop frequency $f_b$ and chirp length $\tau$, we can
    define a unitless parameter

    \[
    b = \frac{1/f_b - 1/f_a}{\tau}.
    \]

    In the passband, the HFM chirp is then given by

    \[
    s(t) = \exp \left(j2\pi \frac{\ln(1 + b f_a t)}{b} \right)
    \qquad
    0 \leq t \leq \tau,
    \]

    which has a hyperbolic instantaneous frequency $f(t) = f_a / (1 + b f_a t)$. The
    inverse of this is the instantaneous period

    \[
    \frac{1}{f(t)} = \frac{1}{f_a} + bt
    \]

    which is linear, giving rise to the alternative name of linear period modulated
    (LPM) chirp.

    """

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
        f_start
            The start frequency of the chirp.
        f_stop
            The stop frequency of the chirp.
        duration
            The duration of the chirp in seconds.
        rms_spl
            The RMS sound pressure level (decibels relative to 1 micropascal) of the
            signal.
        rms_after_window
            If True, scale the signal to the desired RMS SPL after applying the window.
            If False, scale before applying the window, meaning the windowed signal will
            have a lower RMS SPL. If no window is applied, this will be set to False.
        window
            Plugin specification for a signal window to apply to the samples of the
            signal.

        """
        self.f_start = f_start
        self.f_stop = f_stop
        self._minimum_frequency = min(f_start, f_stop)
        self._maximum_frequency = max(f_start, f_stop)
        self._duration = duration
        self.rms_spl = rms_spl
        self.rms_after_window = rms_after_window and (window is not None)
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

        # Calculate the passband phase, convert to baseband and generate.
        b = ((1 / self.f_stop) - (1 / self.f_start)) / self._duration
        phase = np.log(1 + (b * self.f_start * tv)) / b
        phase -= baseband_frequency * tv
        s[valid] = np.exp(2j * np.pi * phase)

        # Calculate the desired source level in Pascal.
        level = 10 ** (self.rms_spl / 20) * 1e-6

        # Not windowing => the current source level is 1Pa.
        if not self.rms_after_window:
            s *= level

        if self.window is not None:
            s *= self.window.get_samples(t, self._duration, fill_value=0)

        if self.rms_after_window:
            # Don't assume the sample times are sorted. argsort() gives us the indices
            # to select to get a sorted array.
            idx = np.argsort(tv)

            # And then sort the valid part of the output.
            t_s = tv[idx]
            s_s = s[valid][idx]

            # Weighting the mean allows for a variable sample spacing.
            dt = np.diff(t_s)
            current = np.sqrt(np.average(np.abs(s_s[1:]) ** 2, weights=dt))

            # Which we can then adjust as desired.
            s *= level / current

        return s

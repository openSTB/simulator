# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Windows that can be applied to transmitted signals.

This module is designed to be similar in nature to the `scipy.signal.windows` module in
SciPy. The function `get_window_func` looks up a window function by name. In SciPy, the
windows are created from the number of coefficients. Here, the windows are created from
the times of the signal samples and the signal duration, and values outside the signal
are set to a fill value.

"""

import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import SignalWindow

_ = translations.load("openstb.simulator").gettext


class GeneralisedCosineWindow(SignalWindow):
    r"""A generalised cosine window.

    The generalised cosine window is the weighted sum of a series of harmonic cosines.
    Many common windows can be implemented as generalised cosine windows.

    Notes
    -----
    For a list of coefficients a of length K, the generalised cosine window is given by

    .. math::

        w(t) = \sum_{k=0}^{K-1} (-1)^k a[k] \cos\frac{2\pi k t}{T}

    for :math:`0 \leq t \leq T` where :math:`T` is the signal duration.

    """

    def __init__(self, coefficients: list[float]):
        """
        Parameters
        ----------
        coefficients : list of floats
            The coefficients of the cosines.

        """
        if not coefficients:
            raise ValueError(
                _("at least one coefficient needed for a generalised cosine window")
            )
        self.coefficients = coefficients

    def get_samples(
        self, t: ArrayLike, duration: float, fill_value: float = 0
    ) -> np.ndarray:
        # Find where the time is within the signal duration.
        t = np.asarray(t)
        valid = (t >= 0) & (t <= duration)

        # Compute the sum for the valid times.
        sgn = 1
        arg = 2 * np.pi * t[valid] / duration
        w_valid = np.zeros_like(arg, dtype=float)
        for k, a_k in enumerate(self.coefficients):
            w_valid += sgn * a_k * np.cos(k * arg)
            sgn = -sgn

        # And use that to create the final array.
        w = np.full_like(t, fill_value, dtype=float)
        w[valid] = w_valid
        return w


class BlackmanWindow(GeneralisedCosineWindow):
    """A Blackman window.

    The Blackman window is a three-term generalised cosine window with coefficients
    a0=0.42, a1=0.5 and a2 = 0.08.

    """

    def __init__(self):
        super().__init__(coefficients=[0.42, 0.5, 0.08])


class BlackmanHarrisWindow(GeneralisedCosineWindow):
    """A Blackman-Harris window.

    The Blackman-Harris window is a four-term generalised cosine window with
    coefficients a0=0.35875, a1=0.48829, a2=0.14128 and a3 = 0.01168.

    """

    def __init__(self):
        super().__init__(coefficients=[0.35875, 0.48829, 0.14128, 0.01168])


class HammingWindow(GeneralisedCosineWindow):
    """A Hamming window.

    The Hamming window is a two-term generalised cosine window. The first term is raised
    above the axis, resulting in a window that does not reach zero but places a
    zero-crossing in a position which cancels the first sidelobe of a Hann (raised
    cosine) window.

    """

    def __init__(self, mode: str = "optimal"):
        """
        Parameters
        ----------
        mode : {"optimal", "original", "scipy"}
            If "optimal", use the optimal cosine coefficients a0=0.53836 and a1=0.46164
            which result in the lowest sidelobes. If "original", use Hamming's original
            parameters a0=25/46 and a1=21/46. If "scipy", use the same parameters as
            `scipy.signal.window.hamming` a0=0.54 and a1=0.46.

        """
        if mode == "optimal":
            a0 = 0.53836
        elif mode == "original":
            a0 = 25 / 46
        elif mode == "scipy":
            a0 = 0.54
        else:
            raise ValueError(_("unknown Hamming mode {name:s}").format(name=mode))
        super().__init__(coefficients=[a0, 1 - a0])


class HannWindow(GeneralisedCosineWindow):
    """A Hann window.

    Also known as a raised cosine window or von Hann window, and sometimes incorrectly
    as a Hanning window, presumably due to the similarity with the Hamming window.

    """

    def __init__(self):
        super().__init__(coefficients=[0.5, 0.5])


class NuttallWindow(GeneralisedCosineWindow):
    """A Nuttall window.

    The Nuttall (sometimes called the Blackman-Nutall) window is a four-term generalised
    cosine window with coefficients a0=0.3635819,, a1=0.4891775, a2=0.1365995 and
    a3=0.0106411.

    """

    def __init__(self):
        super().__init__(coefficients=[0.3635819, 0.4891775, 0.1365995, 0.0106411])


class TukeyWindow(SignalWindow):
    """A Tukey window.

    The Tukey or cosine-tapered window has a taper formed by half a cosine at either end
    of the window covering a given fraction of the window.

    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : float
            The fraction of the window that is tapered, 0 ≤ alpha ≤ 1. Note that this is
            the total fraction used for tapering; alpha/2 of either end of the window is
            tapered. alpha = 0 results in a rectangular window and alpha = 1 in a Hann
            window.

        """
        if not 0 <= alpha <= 1:
            raise ValueError(_("alpha must be between 0 and 1 for a Tukey window"))
        self.alpha = alpha

    def get_samples(
        self, t: ArrayLike, duration: float, fill_value: float = 0
    ) -> np.ndarray:
        # alpha is total taper, half at each end.
        T_taper = duration * self.alpha / 2.0

        # Find the three regions of the window.
        t = np.asarray(t)
        start = (t >= 0) & (t < T_taper)
        body = (t >= T_taper) & (t <= (duration - T_taper))
        end = (t > (duration - T_taper)) & (t <= duration)

        # Form the window. The start taper is cos(-π) to cos(0) scaled to rise from 0 to
        # 1, and the end taper is cos(0) to cos(π) scaled to fall from 1 to 0.
        window = np.full_like(t, fill_value, dtype=float)
        window[start] = 0.5 * (1 + np.cos(np.pi * (t[start] / T_taper - 1)))
        window[body] = 1.0
        window[end] = 0.5 * (
            1 + np.cos(np.pi * (t[end] - (duration - T_taper)) / T_taper)
        )

        return window

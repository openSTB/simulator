# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Windows that can be applied to transmitted signals.

This module is designed to be similar in nature to the `scipy.signal.windows` module in
SciPy. The function `get_window_func` looks up a window function by name. In SciPy, the
windows are created from the number of coefficients. Here, the windows are created from
the times of the signal samples and the signal duration, and values outside the signal
are set to a fill value.

Note that these windows are not plugins, but are designed to be called from within
signal plugins to simplify their implementation.

"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import domain_translator


_ = domain_translator("openstb.simulator", plural=False)
_n = domain_translator("openstb.simulator", plural=True)


class SignalWindowFunc(Protocol):
    """Typing protocol for window functions."""

    def __call__(  # pragma:no cover
        self, t: ArrayLike, duration: float, fill: float = 0, **params: Any
    ) -> np.ndarray: ...


def get_window_func(name: str) -> SignalWindowFunc:
    """Load a signal window function.

    Parameters
    ----------
    name : str
        The name of the window.

    Returns
    -------
    callable
        A function generating samples of the window.

    """
    lname = name.strip().lower()
    func = {
        "blackman": blackman_window,
        "blackman_harris": blackman_harris_window,
        "generalised_cosine": generalised_cosine_window,
        "hamming": hamming_window,
        "hann": hann_window,
        "nuttall": nuttall_window,
        "tukey": tukey_window,
    }.get(lname, None)

    if func is None:
        raise ValueError(_("unknown window {name}").format(name=name))
    return func


def generalised_cosine_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    r"""A generalised cosine window.

    The generalised cosine window is the weighted sum of a series of harmonic cosines.
    Many common windows can be implemented as generalised cosine windows.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.
    coefficients : list of floats
        The coefficients of the cosines.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    Notes
    -----
    For a list of coefficients a of length K, the generalised cosine window is given by

    .. math::

        w(t) = \sum_{k=0}^{K-1} (-1)^k a[k] \cos\frac{2\pi k t}{T}

    for :math:`0 \leq t \leq T` where :math:`T` is the signal duration.

    """
    coeff = params.pop("coefficients", None)
    if coeff is None:
        raise ValueError(_("generalised cosine window needs a list of coefficients"))
    if params:
        raise ValueError(
            _n(
                "unexpected parameter given to generalised cosine window",
                "unexpected parameters given to generalised cosine window",
                len(params),
            )
        )

    # Find where the time is within the signal duration.
    t = np.asarray(t)
    valid = (t >= 0) & (t <= duration)

    # Compute the sum for the valid times.
    sgn = 1
    arg = 2 * np.pi * t[valid] / duration
    w_valid = np.zeros_like(arg, dtype=float)
    for k, a_k in enumerate(coeff):
        w_valid += sgn * a_k * np.cos(k * arg)
        sgn = -sgn

    # And use that to create the final array.
    w = np.full_like(t, fill, dtype=float)
    w[valid] = w_valid
    return w


def blackman_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    """A Blackman window.

    The Blackman window is a three-term generalised cosine window with coefficients
    a0=0.42, a1=0.5 and a2 = 0.08.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    if params:
        raise ValueError(_("Blackman window does not accept any parameters"))
    return generalised_cosine_window(t, duration, fill, coefficients=[0.42, 0.5, 0.08])


def blackman_harris_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    """A Blackman-Harris window.

    The Blackman-Harris window is a four-term generalised cosine window with
    coefficients a0=0.35875, a1=0.48829, a2=0.14128 and a3 = 0.01168.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    if params:
        raise ValueError(_("Blackman-Harris window does not accept any parameters"))
    return generalised_cosine_window(
        t, duration, fill, coefficients=[0.35875, 0.48829, 0.14128, 0.01168]
    )


def hamming_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    """A Hamming window.

    The Hamming window is a two-term generalised cosine window. The first term is raised
    above the axis, resulting in a window that does not reach zero but places a
    zero-crossing in a position which cancels the first sidelobe of a Hann (raised
    cosine) window.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.
    optimal : Boolean, default True
        If True, use the optimal parameters a0=0.53836 and a1=0.46164 which result in
        the lowest sidelobes. If False, use Hamming's original parameters a0=25/46 and
        a1=21/46.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    optimal = params.pop("optimal", True)
    if optimal:
        a0 = 0.53836
    else:
        a0 = 25 / 46
    if params:
        raise ValueError(
            _n(
                "unexpected parameter given to Hamming window",
                "unexpected parameters given to Hamming window",
                len(params),
            )
        )

    return generalised_cosine_window(t, duration, fill, coefficients=[a0, 1 - a0])


def hann_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    """A Hann window.

    Also known as a raised cosine window or von Hann window, and sometimes incorrectly
    as a Hanning window, presumably due to the similarity with the Hamming window.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    if params:
        raise ValueError(_("Hann window does not accept any parameters"))
    return generalised_cosine_window(t, duration, fill, coefficients=[0.5, 0.5])


def nuttall_window(t: ArrayLike, duration: float, fill: float = 0, **params):
    """A Nuttall window.

    The Nuttall (sometimes called the Blackman-Nutall) window is a four-term generalised
    cosine window with coefficients a0=0.3635819,, a1=0.4891775, a2=0.1365995 and
    a3=0.0106411.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    if params:
        raise ValueError(_("Nuttall window does not accept any parameters"))
    return generalised_cosine_window(
        t, duration, fill, coefficients=[0.3635819, 0.4891775, 0.1365995, 0.0106411]
    )


def tukey_window(
    t: ArrayLike, duration: float, fill: float = 0, **params
) -> np.ndarray:
    """A Tukey window.

    The Tukey or cosine-tapered window has a taper formed by half a cosine at either end
    of the window covering a given fraction of the window.

    Parameters
    ----------
    t : array-like
        The time instances to calculate the window at.
    duration : float
        The duration of the signal.
    fill : float
        The value to use to fill samples outside the bounds of the signal.
    alpha : float
        The fraction of the window that is tapered, 0 ≤ alpha ≤ 1. Note that this is the
        total fraction used for tapering; alpha/2 of either end of the window is
        tapered. alpha = 0 results in a rectangular window and alpha = 1 in a Hann
        window.

    Returns
    -------
    window : numpy.ndarray
        An array of floats containing the window values, or ``fill`` outside the bounds
        of the signal. This will have the same shape as ``t``.

    """
    alpha = params.pop("alpha", None)
    if alpha is None:
        raise ValueError(_("Tukey window requires alpha parameter"))
    if not 0 <= alpha <= 1:
        raise ValueError(_("alpha must be between 0 and 1 for a Tukey window"))
    if params:
        raise ValueError(
            _n(
                "unexpected parameter given to Tukey window",
                "unexpected parameters given to Tukey window",
                len(params),
            )
        )

    # alpha is total taper, half at each end.
    T_taper = duration * alpha / 2.0

    # Find the three regions of the window.
    t = np.asarray(t)
    start = (t >= 0) & (t < T_taper)
    body = (t >= T_taper) & (t <= (duration - T_taper))
    end = (t > (duration - T_taper)) & (t <= duration)

    # Form the window. The start taper is cos(-π) to cos(0) scaled to rise from 0 to 1,
    # and the end taper is cos(0) to cos(π) scaled to fall from 1 to 0.
    window = np.full_like(t, fill, dtype=float)
    window[start] = 0.5 * (1 + np.cos(np.pi * (t[start] / T_taper - 1)))
    window[body] = 1.0
    window[end] = 0.5 * (1 + np.cos(np.pi * (t[end] - (duration - T_taper)) / T_taper))

    return window

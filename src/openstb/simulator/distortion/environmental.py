# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult

_ = translations.load("openstb.simulator").gettext


class AnslieMcColmAttenuation(Distortion):
    r"""Attenuation of acoustic energy using the simplified Anslie-McColm model.

    The model presented by Anslie and McColm [1]_ models the attenuation due to the
    chemical relaxation of both boric acid and magnesium sulphate, and the viscous drag
    of the water. Along with the frequency and operating depth, the temperature,
    salinity and pH of the water are parameters. For common values, only the salinity
    has a significant affect. Broadly speaking, the chemical relaxation of boric acid
    dominates below 1kHz, the chemical relaxation of magnesium sulphate dominates from
    10kHz to 100kHz and the viscous drag dominates above 200kHz.

    Notes
    -----
    The model in the paper takes the frequency in kHz and the depth in kilometres. It
    characterises the chemical relaxations in terms of an attenuation coefficient and a
    relaxation frequency. Boric acid has a relaxation frequency

    .. math:: f_1 = 0.78\sqrt{(S/35)}\,e^{T/26}

    and a coefficient

    .. math:: A = 0.106 e^{(\mathrm{pH} - 8)/0.56}

    for salinity S and temperature T. Similarly, the relaxation frequency of magnesium
    sulphate is

    .. math:: f_2 = 42e^{T/17}

    and its attenuation coefficient is

    .. math:: B = 0.52 \left(1 + \frac{T}{43}\right) \left(\frac{S}{35}\right) e^{-D/6}

    for depth D. Finally, the viscous drag has an attenuation coefficient of

    .. math:: C = 0.00049 e^{-(T/27 + D/17)}.

    The total attenuation, in dB/km, is then given by

    .. math:: \alpha_a(f) = \frac{Af_1f^2}{f_1^2 + f^2} +
                            \frac{Bf_2f^2}{f_2^2 + f^2} + Cf^2.

    References
    ----------
    .. [1] Ainslie, M. A. and McColm, J. G, "A simplified formula for viscous and
           chemical absorption in sea water". The Journal of the Acoustical Society of
           America, volume 103 number 3, pp. 1671-1672, 1998.

    """

    def __init__(self, frequency: str, pH: float = 8.0):
        """
        Parameters
        ----------
        frequency : {"min", "centre", "max", "all}
            Which frequency or frequencies to calculate the attenuation at. If "min" or
            "max", use the minimum or maximum frequency in the signal, respectively.
            When "centre", use the centre frequency, i.e., (min + max) / 2. If "all",
            calculate the attenuation at each frequency being sampled in the simulation.
        pH : float
            The pH of the water.

        """
        if frequency == "min":
            self._mode = 0
        elif frequency == "max":
            self._mode = 1
        elif frequency == "centre":
            self._mode = 2
        elif frequency == "all":
            self._mode = 3
        else:
            raise ValueError(
                _("unknown value for frequency: {value}").format(value=frequency)
            )

        self.ph = pH

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
        # The formulas in the paper use kHz. Ensure we have a 3d array with frequency
        # on the middle axis.
        if self._mode == 0:
            fk = np.array(signal_frequency_bounds[0] / 1000.0).reshape(1, 1, 1)
        elif self._mode == 1:
            fk = np.array(signal_frequency_bounds[1] / 1000.0).reshape(1, 1, 1)
        elif self._mode == 2:
            fk = np.array(
                (signal_frequency_bounds[0] + signal_frequency_bounds[1]) / 2000.0
            ).reshape(1, 1, 1)
        else:
            fk = (np.array(f) / 1000.0)[np.newaxis, :, np.newaxis]

        # And depth in kilometres.
        depth = tt_result.tx_position[2] / 1000.0

        # Use the environment plugin to determine salinity and temperature at transmit.
        salinity = environment.salinity(ping_time, tt_result.tx_position)
        temp = environment.temperature(ping_time, tt_result.tx_position)

        # Calculate relaxation constants.
        A = 0.106 * np.exp((self.ph - 8) / 0.56)
        B = 0.52 * (1 + temp / 43) * (salinity / 35) * np.exp(-depth / 6)
        C = 0.00049 * np.exp(-(temp / 27 + depth / 17))

        # Calculate relaxation frequencies.
        f1 = 0.78 * np.sqrt(salinity / 35) * np.exp(temp / 26)
        f2 = 42 * np.exp(temp / 17)

        # Calculate the components.
        fsq = fk**2
        boric = A * fsq * f1 / (f1**2 + fsq)
        magsulphate = B * fsq * f2 / (f2**2 + fsq)
        viscous = C * fsq

        # And the total attenuation, in dB/km.
        total = boric + magsulphate + viscous

        # Calculate the attenuation in dB for the echoes.
        r = tt_result.tx_path_length + tt_result.rx_path_length
        atten_db = total * r[:, np.newaxis, :] / 1000

        # And convert back to amplitude.
        return np.array(S) * 10 ** (-atten_db / 20)


class GeometricSpreading(Distortion):
    """Amplitude distortion due to geometric spreading of the signal."""

    def __init__(self, power: float):
        """
        Parameters
        ----------
        power : float
            The power to use when calculating a one-way spreading loss a = 1/r^power.
            This is applied to the amplitude, so a power of 1 corresponds to spherical
            spreading (1/r in amplitude, 1/r^2 in intensity) and a power of 0.5
            corresponds to cylindrical spreading.

        """
        self.power = power

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
        txscale = 1 / (tt_result.tx_path_length**self.power)
        rxscale = 1 / (tt_result.rx_path_length**self.power)
        return np.array(S) * (txscale * rxscale)[:, np.newaxis, :]

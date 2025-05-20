# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import Distortion, Environment, TravelTimeResult
from openstb.simulator.util import rotate_elementwise

_ = translations.load("openstb.simulator").gettext


class RectangularBeampattern(Distortion):
    """Scaling due to the beampattern of an ideal rectangular aperture."""

    def __init__(
        self,
        width: float,
        height: float,
        transmit: bool,
        receive: bool,
        frequency: str,
        horizontal: bool = True,
        vertical: bool = True,
    ):
        """
        Parameters
        ----------
        width, height : float
            The size of the aperture in metres.
        transmit, receive: Boolean
            Whether to apply this for the transmited and/or received signal. At least
            one of these must be True.
        frequency : {"min", "centre", "max", "all}
            Which frequency or frequencies to calculate the attenuation at. If "min" or
            "max", use the minimum or maximum frequency in the signal, respectively.
            When "centre", use the centre frequency, i.e., (min + max) / 2. If "all",
            calculate the attenuation at each frequency being sampled in the simulation.
        horizontal, vertical : Boolean, default True
            Whether to include the horizontal and/or vertical components of the
            beampattern in the scaling. At least one of these must be true.


        """
        # Check the size.
        if not width > 0:
            raise ValueError(_("width of aperture must be positive"))
        if not height > 0:
            raise ValueError(_("height of aperture must be positive"))
        self.width = width
        self.height = height

        # Check when to apply.
        self.transmit = transmit
        self.receive = receive
        if not self.transmit and not self.receive:
            raise ValueError(_("at least one of transmit and receive must be true"))

        # Which frequency to evaluate at.
        if frequency == "min":
            self._freqmode = 0
        elif frequency == "max":
            self._freqmode = 1
        elif frequency == "centre":
            self._freqmode = 2
        elif frequency == "all":
            self._freqmode = 3
        else:
            raise ValueError(
                _("unknown value for frequency: {value}").format(value=frequency)
            )

        # Components to include.
        self.horizontal = horizontal
        self.vertical = vertical
        if not self.horizontal and not self.vertical:
            raise ValueError(_("at least one of horizontal and vertical must be true"))

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
        # Use the sound speed at transmit.
        sound_speed = environment.sound_speed(ping_time, tt_result.tx_position)
        sound_speed = np.array(sound_speed).reshape(1, 1, 1)

        # Generate a wavelength array depending on the frequency mode.
        if self._freqmode == 0:
            # Minimum signal frequency.
            wl = sound_speed / signal_frequency_bounds[0]
        elif self._freqmode == 1:
            # Maximum signal frequency:
            wl = sound_speed / signal_frequency_bounds[1]
        elif self._freqmode == 2:
            # Centre signal frequency
            wl = sound_speed / (np.sum(signal_frequency_bounds) / 2)
        else:
            # All frequencies being sampled.
            wl = sound_speed / np.array(f)[np.newaxis, :, np.newaxis]

        # In the following, we rotate the vectors back into the transducer coordinate
        # system (so x is the transducer normal). In the case of the receiver, we also
        # negate the components: the vector points into the transducer in this case, and
        # we want it to point away from it.

        distorted = np.array(S)

        # Evaluate for the transmitter.
        if self.transmit:
            tx_vector = rotate_elementwise(
                ~tt_result.tx_orientation,
                tt_result.tx_vector[np.newaxis, np.newaxis, :],
            )
            distorted = distorted * self._eval(wl, tx_vector)

        # And for the receivers.
        if self.receive:
            rx_vector = rotate_elementwise(
                ~tt_result.rx_orientation, -tt_result.rx_vector[:, np.newaxis, ...]
            )
            distorted = distorted * self._eval(wl, rx_vector)

        return distorted

    def _eval(self, wavelength: np.ndarray, direction: np.ndarray) -> np.ndarray:
        if self.horizontal and self.vertical:
            return np.sinc(self.width * direction[..., 1] / wavelength) * np.sinc(
                self.height * direction[..., 2] / wavelength
            )

        if self.horizontal:
            return np.sinc(self.width * direction[..., 1] / wavelength)

        return np.sinc(self.height * direction[..., 2] / wavelength)

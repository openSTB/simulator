# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike

from openstb.i18n.support import domain_translator
from openstb.simulator import abc


_ = domain_translator("openstb.simulator", plural=False)


class Rectangular(abc.Beampattern):
    """The beampattern from an ideal rectangular aperture."""

    def __init__(self, width: float, height: float):
        """
        Parameters
        ----------
        width, height : float
            The size of the aperture in metres. Setting one of the components to zero
            disables the beampattern in that direction, i.e., width=0 will disable the
            horizontal beampattern and height=0 will disable the vertical beampattern.
            At least one of these must be non-zero.

        """
        if width < 0:
            raise ValueError(_("width of aperture cannot be negative"))
        if height < 0:
            raise ValueError(_("height of aperture cannot be negative"))
        self.width = width
        self.height = height

        # Figure out which directions to evaluate.
        eval_w = not np.isclose(width, 0)
        eval_h = not np.isclose(height, 0)
        if eval_w and eval_h:
            self.mode = 2
        elif eval_w:
            self.mode = 0
        elif eval_h:
            self.mode = 1
        else:
            raise ValueError(_("cannot have width and height both be zero"))

    def evaluate(self, wavelength: ArrayLike, direction: ArrayLike) -> np.ndarray:
        wavelength = np.asarray(wavelength)
        direction = np.asarray(direction)

        # Horizontal only.
        if self.mode == 0:
            return np.sinc(self.width * direction[..., 1] / wavelength)

        # Vertical only.
        if self.mode == 1:
            return np.sinc(self.height * direction[..., 2] / wavelength)

        # Both directions.
        return np.sinc(self.width * direction[..., 1] / wavelength) * np.sinc(
            self.height * direction[..., 2] / wavelength
        )

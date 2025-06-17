# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import Distortion, Transducer
from openstb.simulator.plugin.loader import distortion
from openstb.simulator.types import PluginOrSpec

_ = translations.load("openstb.simulator").gettext


class GenericTransducer(Transducer):
    """A generic transducer."""

    def __init__(
        self,
        position: ArrayLike,
        orientation: ArrayLike | quaternionic.QArray,
        beampattern: PluginOrSpec[Distortion] | None = None,
    ):
        """
        Parameters
        ----------
        position
            The position of the transducer relative to the system origin.
        orientation
            The orientation of the transducer boresight relative to the x axis of the
            system.
        beampattern
            The distortion model of the beampattern.

        """
        self._position = np.array(position, dtype=float)
        if self._position.shape != (3,):
            raise ValueError(_("position should be a 3-element vector"))

        try:
            self._orientation = quaternionic.array(orientation)
        except ValueError:
            raise ValueError(_("transducer orientation must be a valid quaternion"))
        if self._orientation.shape != (4,):
            raise ValueError(_("transducer orientation must be a single quaternion"))

        if beampattern is None:
            self._distortion = []
        else:
            self._distortion = [distortion(beampattern)]

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def orientation(self) -> quaternionic.QArray:
        return self._orientation

    @property
    def distortion(self) -> list[Distortion]:
        return self._distortion

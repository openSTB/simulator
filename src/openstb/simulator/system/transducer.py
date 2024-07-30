# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import domain_translator
from openstb.simulator.plugin.abc import ScaleFactor, Transducer
from openstb.simulator.plugin.loader import PluginSpec, scale_factor


_ = domain_translator("openstb.simulator", plural=False)


class GenericTransducer(Transducer):
    def __init__(
        self,
        position: ArrayLike,
        orientation: ArrayLike | quaternionic.QArray,
        beampattern: PluginSpec | None = None,
    ):
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
            self._scale_factors = []
        else:
            self._scale_factors = [scale_factor(beampattern)]

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def orientation(self) -> quaternionic.QArray:
        return self._orientation

    @property
    def scale_factors(self) -> list[ScaleFactor]:
        return self._scale_factors

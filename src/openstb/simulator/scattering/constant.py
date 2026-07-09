# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Constant scattering strength."""

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin.abc import Environment, ScatteringModel


class ConstantScattering(ScatteringModel):
    """Interface with a constant scattering strength.

    This models the intensity of the scattered wave as the intensity of the incident
    wave multiplied by a constant scale factor.

    """

    def __init__(self, scale_factor: float):
        """
        Parameters
        ----------
        scale_factor
            Multiplicative scale factor to get the scattered intensity from the incident
            intensity.

        """
        self.scale_factor = scale_factor

    def apply(
        self,
        f: ArrayLike,
        S: ArrayLike,
        time: ArrayLike,
        position: ArrayLike,
        normal: ArrayLike,
        incident_vector: ArrayLike,
        scattering_vector: ArrayLike,
        environment: Environment,
        baseband_frequency: float,
        signal_frequency_bounds: tuple[float, float],
    ) -> np.ndarray:
        return np.asarray(S) * np.sqrt(self.scale_factor)

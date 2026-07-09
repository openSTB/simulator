# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Lambertian scattering."""

import numpy as np
from numpy.typing import ArrayLike

from openstb.simulator.plugin.abc import Environment, ScatteringModel


class LambertianScattering(ScatteringModel):
    r"""Lambertian scattering from an interface.

    This assumes omnidirectional scattering from the interface. The strength of the
    scattered intensity is I₀ cos ϴ where I₀ is the incident energy and ϴ the angle
    between the incident vector and the normal of the interface.

    """

    def __init__(self, scale_factor: float = 1):
        """
        Parameters
        ----------
        scale_factor
            An additional multiplicative scale factor to include, for example to model a
            material-dependent scattering strength.

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
        # The incident intensity is scaled by the cosine of the angle between the
        # surface normal and the incident wave. As we modify the amplitude, we have to
        # take the square root of this.
        fac = self.scale_factor * np.abs(np.vecdot(normal, incident_vector, axis=-1))  # type:ignore[call-overload]
        return np.asarray(S) * np.sqrt(fac)

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
from numpy.typing import ArrayLike
import quaternionic

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import PointTargets

_ = translations.load("openstb.simulator").gettext


class RandomPointRect(PointTargets):
    """A set of points uniformly distributed within a rectangle.

    The points are placed uniformly at a given density within a rectangle of a set size
    in the x-y plane, and are then translated and rotated to the desired position and
    orientation.

    """

    def __init__(
        self,
        seed: int,
        Dx: float,
        Dy: float,
        centre: ArrayLike,
        normal: ArrayLike,
        point_density: float,
        reflectivity: str | float,
    ):
        """
        Parameters
        ----------
        seed : positive int
            Seed for the random number generator.
        Dx, Dy : float
            Size (in metres) of the rectangle in the x-y plane prior to transformation.
        centre : array-like
            A 3-element vector giving the centre of the final rectangle in global
            coordinates.
        normal : array-like
            A 3-element vector giving the normal of the final rectangle. Remember that
            the z axis points down, so an upwards-facing rectangle would have a normal
            with a negative z value. The vector does not need to have unit length, but
            cannot have zero length.
        point_density : float
            The density of the points in points per square metre.
        reflectivity : float, {"omnidirectional", "hemispherical"}
            The reflectivity of the point targets. This is a amplitude scaling factor
            applied to the incident pulse to get the scattered pulse. The special values
            "omnidirectional" and "hemispherical" model omnidirectional (uniform
            scattering over the sphere) and hemispherical (uniform scattering over the
            hemisphere facing the sonar) scattering and correspond to 1/4pi and 1/2pi
            respectively.

        """
        self.rng = np.random.default_rng(seed)
        self.Dx = Dx
        self.Dy = Dy
        self.point_density = point_density

        # Check the centre is valid.
        self.centre = np.array(centre)
        if self.centre.shape != (3,):
            raise ValueError(
                _("centre position of rectangle must have exactly 3 values")
            )

        # Check and scale the given normal.
        normal = np.array(normal, dtype=float)
        if normal.shape != (3,):
            raise ValueError(_("normal of rectangle must have exactly 3 values"))
        nlen = np.linalg.norm(normal)
        if np.isclose(nlen, 0):
            raise ValueError(_("length of rectangle normal cannot be zero"))
        normal /= nlen

        # Convert to a quaternion we can use for rotating the generated points.
        dp = np.dot(normal, [0, 0, -1])
        if np.isclose(dp, 1):
            self._ori = quaternionic.array([1, 0, 0, 0])
        elif np.isclose(dp, -1):
            self._ori = quaternionic.array([0, 1, 0, 0])
        else:
            angle = np.arccos(dp)
            axis = np.cross(normal, [0, 0, -1])
            axis /= np.linalg.norm(axis)
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            self._ori = quaternionic.array([c, s * axis[0], s * axis[1], s * axis[2]])

        # Handle the specific cases for the reflectivity.
        if isinstance(reflectivity, str):
            rlower = reflectivity.lower()
            if rlower == "omnidirectional":
                self._reflectivityval = 1 / (4 * np.pi)
            elif rlower == "hemispherical":
                self._reflectivityval = 1 / (2 * np.pi)
            else:
                raise ValueError(
                    _("unexpected value '{value}' for reflectivity").format(
                        value=reflectivity
                    )
                )

        # Assume a specified value.
        else:
            self._reflectivityval = reflectivity

    def prepare(self):
        # Figure out the number of points needed to meet the density.
        area = self.Dx * self.Dy
        self._len = int(np.round(area * self.point_density))

        # Place them uniformly within the xy plane.
        position = np.empty((self._len, 3), dtype=float)
        position[:, 0] = self.rng.uniform(-self.Dx / 2, self.Dx / 2, self._len)
        position[:, 1] = self.rng.uniform(-self.Dy / 2, self.Dy / 2, self._len)
        position[:, 2] = 0

        # Rotate and shift to the desired positions.
        self._position = self.centre + self._ori.rotate(position)

        # Fill an array with the reflectivity value.
        self._reflectivity = np.full(self._len, self._reflectivityval, dtype=float)

    def __len__(self) -> int:
        return self._len

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def reflectivity(self) -> np.ndarray:
        return self._reflectivity


class SinglePoint(PointTargets):
    """A single point target."""

    def __init__(self, position: ArrayLike, reflectivity: float):
        """
        Parameters
        ----------
        position : array-like
            The position of the point target in global coordinates.
        reflectivity : float
            The reflectivity of the target (the fraction of incident energy that will
            scatter back to the receiver).

        """
        self._position = np.array(position).reshape(1, 3)
        self._reflectivity = np.array(reflectivity).reshape(
            1,
        )

    def __len__(self) -> int:
        return 1

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def reflectivity(self) -> np.ndarray:
        return self._reflectivity

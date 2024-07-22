# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: CC0-1.0

# This is a basic stub which tells a static type checker about the SciPy interpolation
# methods we use.

import numpy as np

class PchipInterpolator:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis: int = 0,
        extrapolate: bool | None = None,
    ): ...
    def __call__(
        self, x: np.ndarray, nu: int = 0, extrapolate: bool | None = None
    ) -> np.ndarray: ...

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: CC0-1.0

# This is a basic stub file which allows mypy to extract type information about the
# quaternionic functions we use. This is not an accurate reflection of the design of
# quaternionic, which can use underlying arrays of different dtypes etc which
# complicates static typing. Since we only use a small subset of what it can do, this
# should suffice until full typing support is available.

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

class QArray:
    def rotate(self, vecs: ArrayLike) -> np.ndarray: ...
    def __getitem__(self, idx: Any) -> QArray: ...
    def __invert__(self) -> QArray: ...
    def __mul__(self, other: QArray) -> QArray: ...

    ndarray: np.ndarray
    w: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    scalar: np.ndarray
    vector: np.ndarray
    mag2: np.ndarray
    shape: tuple[int]

class _qarray_stub:
    def __call__(self, x: ArrayLike | QArray) -> QArray: ...
    def from_rotation_vector(self, x: ArrayLike) -> QArray: ...

array = _qarray_stub()

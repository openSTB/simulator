# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: CC0-1.0

# This is a basic stub file which allows mypy to extract type information about the
# quaternionic functions we use. This is not an accurate reflection of the design of
# quaternionic, which can use underlying arrays of different dtypes etc which
# complicates static typing. Since we only use a small subset of what it can do, this
# should suffice until full typing support is available.

import numpy as np

class QArray(np.ndarray):
    def rotate(self, vecs: np.ndarray) -> np.ndarray: ...

class _qarray_stub:
    def __call__(self, x: np.ndarray) -> QArray: ...
    def from_rotation_vector(self, x: np.ndarray) -> QArray: ...

array = _qarray_stub()
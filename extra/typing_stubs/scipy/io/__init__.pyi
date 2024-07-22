# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: CC0-1.0

# A basic stub which tells a static type checker about the SciPy IO methods we use.

import os
from typing import BinaryIO, Literal

def savemat(
    file_name: os.PathLike[str] | str | BinaryIO,
    mdict: dict,
    appendmat: bool = True,
    format: Literal["5"] | Literal["4"] = "5",
    long_field_names: bool = False,
    do_compression: bool = False,
    oned_as: Literal["row"] | Literal["column"] = "row",
): ...

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import typing

# The underlying array type of a quaternionic array is not fixed (it can be different
# dtypes, or even other array classes like a SymPy array for symbolic quaternion
# analysis). As such, the types are dynamically created. For static typing, we have a
# stub (see utils/typing_stubs/quaternionic.pyi) specifying the pieces of the interface
# we use. This uses a shorthand type QArray; we need to make that available in the
# runtime interface so normal interpreters can access it.
if not typing.TYPE_CHECKING:  # pragma:no cover
    import quaternionic

    quaternionic.QArray = quaternionic.array

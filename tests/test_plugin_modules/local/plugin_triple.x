# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent


class TriplePlugin:
    def __init__(self, offset=0, scale=1):
        self.offset = offset
        self.scale = scale

    def get_value(self, initial=1):
        return self.offset + 2 * initial * self.scale

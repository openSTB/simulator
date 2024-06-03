# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent


class NegatingPlugin:
    def __init__(self, offset=0, scale=1):
        self.offset = offset
        self.scale = scale

    def get_value(self, initial=1):
        return self.offset + -initial * self.scale

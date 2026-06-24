# class: ATestPlugin
# name: plugin_a
# version: 1.0

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent


class ATestPlugin:
    def __init__(self, offset, scale=1):
        self.offset = offset
        self.scale = scale

    def get_value(self, initial=1):
        return self.offset + initial * self.scale

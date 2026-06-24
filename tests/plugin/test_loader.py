# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import pytest

from openstb.simulator.plugin import loader


def test_plugin_loader_missing_file():
    """plugin.loader: can load plugin from file"""

    with pytest.raises(ValueError, match="could not import.+file missing?"):
        loader.load_plugin_class("openstb.simulator.test", "MyPlugin:/non/existent.py")
    with pytest.raises(ValueError, match="could not import.+file missing?"):
        loader.load_plugin_class("openstb.simulator.test", "MyPlugin:../relative.py")

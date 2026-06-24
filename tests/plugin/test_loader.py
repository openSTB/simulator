# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.plugin import loader
from openstb.simulator.system.trajectory import Linear


@pytest.mark.parametrize("load", [False, True])
def test_plugin_loader_list_registered(load: bool):
    """plugin.registered_plugins: finds some included plugins"""
    found = loader.registered_plugins("openstb.simulator.trajectory", load=load)

    found_linear = False
    for details in found:
        if details[0] == "linear":
            assert details[1] == "openstb.simulator.system.trajectory:Linear"
            if load:
                assert details[2] is Linear  # type:ignore[misc]
            found_linear = True

    assert found_linear


def test_plugin_loader_module():
    """plugin.loader: can load plugin from installed module"""
    cls = loader.load_plugin_class(
        "openstb.simulator.testing", "Linear:openstb.simulator.system.trajectory"
    )
    assert cls is Linear

    inst = loader.load_plugin(
        "openstb.simulator.testing",
        {
            "name": "Linear:openstb.simulator.system.trajectory",
            "parameters": {
                "start_position": [0, 0, 0],
                "end_position": [10, 0, 0],
                "speed": 1.2,
            },
        },
    )
    assert isinstance(inst, Linear)
    assert np.allclose(inst.end_position, [10, 0, 0])


def test_plugin_loader_file(tmp_path):
    """plugin.loader: can load plugin from file"""
    fn = tmp_path / "my_plugin.py"
    fn.write_text("""
class MyPlugin:
    def __init__(self, value: int):
        self.value = value

    def double(self) -> int:
        return self.value * 2
""")

    cls = loader.load_plugin_class("openstb.simulator.testing", f"MyPlugin:{fn}")
    assert cls.__name__ == "MyPlugin"
    assert cls.__module__.startswith("openstb.simulator.dynamic_plugins")

    with pytest.raises(AttributeError, match="has no attribute MyPlugin2"):
        cls = loader.load_plugin_class("openstb.simulator.testing", f"MyPlugin2:{fn}")

    inst = loader.load_plugin(
        "openstb.simulator.testing",
        {"name": f"MyPlugin:{fn}", "parameters": {"value": 3}},
    )
    assert inst.double() == 6


def test_plugin_loader_missing_file():
    """plugin.loader: can load plugin from file"""

    with pytest.raises(ValueError, match="could not import.+file missing?"):
        loader.load_plugin_class("openstb.simulator.test", "MyPlugin:/non/existent.py")
    with pytest.raises(ValueError, match="could not import.+file missing?"):
        loader.load_plugin_class("openstb.simulator.test", "MyPlugin:../relative.py")

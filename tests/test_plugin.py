# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Sequence
import importlib.machinery
import importlib.metadata
import os
from pathlib import Path
import sys
import types

import pytest

from openstb.simulator.plugin import loader

# To test our plugin support, we don't want to rely on plugins being installed. In
# particular, we don't want to have to install test plugins as they may have deliberate
# errors designed to check certain pieces of code. Instead, we use a custom meta path
# finder (added to sys.meta_path by the test_plugins fixture) which can load test
# plugins from the tests/test_plugin/modules/registered and
# tests/test_plugin/modules/installed directories. These will be available under the
# openstb_simtestplugin.registered and openstb_simtestplugin.installed namespaces,
# respectively. Each sub-module in the former will also be registered as an entry point
# for the "openstb.simulator.testplugin" group with the name specified in a comment at
# the top of the file.


class PluginTestDistribution(importlib.metadata.Distribution):
    """Distribution information about a registered test plugin.

    Note that this is a severely limited implementation of the Distribution class as we
    don't need much for testing.

    """

    def __init__(self, module: str, entry_point_class: str, regname: str, version: str):
        """
        Parameters
        ----------
        module : str
            Name of the module containing the plugin.
        entry_point_class : str
            Name of the class the entry point should load.
        regname : str
            Registered name of the plugin.
        version : str
            Version string of the plugin.

        """
        self.module = module
        self.cls = entry_point_class
        self.regname = regname

    def read_text(self, filename: str) -> str | None:
        if filename == "entry_points.txt":
            return f"""
            [openstb.simulator.testplugin]
            {self.regname} = openstb_simtestplugin.registered.{self.module}:{self.cls}
            """
        return None

    def locate_file(self, path: str | os.PathLike[str]):
        pass

    @property
    def name(self) -> str:
        return f"openstb_simtestplugin.registered.{self.module}"


class PluginTestFinder(importlib.metadata.DistributionFinder):
    """importlib finder to locate test plugins."""

    def __init__(self):
        self.registered = []

        # Find all 'registered' plugins.
        base = Path(__file__).parent / "test_plugin_modules"
        for fn in (base / "registered").iterdir():
            if fn.suffix == ".py" and fn.stem != "__init__":
                # Default metadata.
                name = fn.stem
                version = "1.0.0"
                ep_class = None

                # Read metadata from the top of the file.
                with fn.open("r") as f:
                    for line in f.readlines():
                        if not line.startswith("#"):
                            break
                        line = line[1:].strip()
                        k, v = line.split(":")
                        k = k.strip()
                        v = v.strip()
                        if k == "class":
                            ep_class = v
                        elif k == "name":
                            name = v
                        elif k == "version":
                            version = v
                        else:
                            raise RuntimeError(f"{fn}: unexpected metadata key {k}")

                if ep_class is None:
                    raise RuntimeError(f"{fn}: missing class metadata key")

                # Create a distribution.
                self.registered.append(
                    PluginTestDistribution(fn.stem, ep_class, name, version)
                )

        # Create a file finder for use in loading the modules.
        self._finder = importlib.machinery.FileFinder(str(base))

    def find_distributions(
        self, context: importlib.metadata.DistributionFinder.Context | None = None
    ) -> list[PluginTestDistribution]:
        # Load a default context.
        if context is None:
            context = importlib.metadata.DistributionFinder.Context()

        # Filter by name if requested.
        if context.name is not None:
            return [dist for dist in self.registered if dist.name == context.name]

        return self.registered

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        # Check the package is our test package.
        package, _, sub = fullname.partition(".")
        if package != "openstb_simtestplugin":
            return None

        # Loading the top-level module. Just point it to the registered directory as we
        # don't need anything from this anyway.
        if not sub:
            spec = self._finder.find_spec("registered")
            if spec is not None:
                spec.name = "openstb_simtestplugin"

        # Loading a submodule.
        else:
            if not (sub.startswith("registered") or sub.startswith("installed")):
                return None
            spec = self._finder.find_spec(sub)
            if spec is not None:
                spec.name = fullname

        return spec


@pytest.fixture(scope="function")
def test_plugins():
    """Fixture to make test plugins available.

    Yields the path to the local plugin directory.

    """
    # Create an instance of our finder if needed, and add it to the meta path.
    if not hasattr(test_plugins, "finder"):
        test_plugins.finder = PluginTestFinder()
    sys.meta_path.append(test_plugins.finder)

    yield Path(__file__).parent / "test_plugin_modules" / "local"

    # Remove the finder and remove any cached plugin modules it found.
    sys.meta_path.remove(test_plugins.finder)
    for name in list(sys.modules.keys()):
        if name.startswith("openstb_simtestplugin"):
            del sys.modules[name]


def test_plugin_registered_empty():
    """plugin.registered_plugins: group with no plugins"""
    # Mostly intended to ensure other tests don't get confused if the user has for some
    # reason installed plugins in this group.
    registered = loader.registered_plugins("openstb.simulator.testplugin")
    assert len(registered) == 0


def test_plugin_registered(test_plugins):
    """plugin.registered_plugins: check basic behaviour"""
    registered = loader.registered_plugins("openstb.simulator.testplugin")
    assert len(registered) == 3
    for record in registered:
        assert len(record) == 2

    registered = loader.registered_plugins("openstb.simulator.testplugin", load=True)
    assert len(registered) == 3
    for record in registered:
        assert len(record) == 3


def test_plugin_load_class(test_plugins):
    """plugin.load_plugin_class: check basic behaviour"""
    cls = loader.load_plugin_class("openstb.simulator.testplugin", "plugin_a")
    assert cls.__name__ == "ATestPlugin"
    inst = cls(scale=2, offset=-1)
    assert inst.get_value(3) == 5

    cls = loader.load_plugin_class(
        "openstb.simulator.testplugin",
        "NegatingPlugin:openstb_simtestplugin.installed.plugin_b",
    )
    assert cls.__name__ == "NegatingPlugin"
    inst = cls(scale=2, offset=-1)
    assert inst.get_value(3) == -7

    cls = loader.load_plugin_class(
        "openstb.simulator.testplugin",
        f"DoublePlugin:{test_plugins / 'plugin_double.py'}",
    )
    assert cls.__name__ == "DoublePlugin"
    inst = cls(scale=2, offset=-1)
    assert inst.get_value(3) == 11


def test_plugin_load_class_missing(test_plugins):
    """plugin.load_plugin_class: check with invalid plugin name"""
    with pytest.raises(ValueError, match="no .+testplugin plugin named 'missing'"):
        loader.load_plugin_class("openstb.simulator.testplugin", "missing")

    with pytest.raises(ModuleNotFoundError):
        loader.load_plugin_class(
            "openstb.simulator.testplugin",
            "Missing:openstb_simtestplugin.installed.missing",
        )

    # If the spec does not refer to an existing file, it is treated as an installed
    # module.
    with pytest.raises(ModuleNotFoundError):
        loader.load_plugin_class(
            "openstb.simulator.testplugin", f"Missing:{test_plugins / 'missing.py'}"
        )


def test_plugin_load_class_multi(test_plugins):
    """plugin.load_plugin_class: check with multiple registered plugins of same name"""
    with pytest.raises(ValueError, match="multiple .+testplugin plugins"):
        loader.load_plugin_class("openstb.simulator.testplugin", "multi")


def test_plugin_load_class_local_err(test_plugins):
    """plugin.load_plugin_class: local plugins with errors"""
    with pytest.raises(SyntaxError, match="expected ':'"):
        loader.load_plugin_class(
            "openstb.simulator.testplugin",
            f"TriplePlugin:{test_plugins / 'plugin_triple.py'}",
        )

    with pytest.raises(ValueError, match="could not create import spec"):
        loader.load_plugin_class(
            "openstb.simulator.testplugin",
            f"TriplePlugin:{test_plugins / 'plugin_triple.x'}",
        )

    with pytest.raises(AttributeError, match="has no attribute DblPlugin"):
        loader.load_plugin_class(
            "openstb.simulator.testplugin",
            f"DblPlugin:{test_plugins / 'plugin_double.py'}",
        )


def test_plugin_load_spec(test_plugins):
    """plugin.load_plugin: check with spec dictionary"""
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        loader.load_plugin("openstb.simulator.testplugin", {"name": "plugin_a"})

    inst = loader.load_plugin(
        "openstb.simulator.testplugin",
        {"name": "plugin_a", "parameters": {"offset": 2, "scale": 5}},
    )
    assert inst.get_value(5) == 27

    # Should pass through an existing plugin instance.
    cls = loader.load_plugin_class("openstb.simulator.testplugin", "plugin_a")
    inst = cls(scale=2, offset=-1)
    assert loader.load_plugin("openstb.simulator.testplugin", inst) is inst

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, overload

from openstb.i18n.support import domain_translator
from openstb.simulator.plugin import abc


_ = domain_translator("openstb.simulator")


@overload
def registered_plugins(
    group: str, load: Literal[True]
) -> list[tuple[str, str, type[abc.Plugin]]]: ...


@overload
def registered_plugins(group: str, load: Literal[False]) -> list[tuple[str, str]]: ...


@overload
def registered_plugins(
    group: str, load: bool
) -> list[tuple[str, str]] | list[tuple[str, str, type[abc.Plugin]]]: ...


def registered_plugins(
    group: str, load: bool = False
) -> list[tuple[str, str]] | list[tuple[str, str, type[abc.Plugin]]]:
    """List registered plugins.

    Parameters
    ----------
    group : str
        The name of the entry point group to list plugins for, e.g.,
        "openstb.simulator.trajectory".
    load : Boolean
        If True, include the loaded plugins in the return. If False, only report the
        name and source of each plugin.

    Returns
    -------
    installed : list
        A list of (name, src) tuples where name is the name the plugin is registered as
        and src is the reference to the module and class implementing the plugin. If
        ``load`` was True, a third entry ``cls`` will be added to each tuple containing
        the loaded plugin class.

    """
    if load:
        return [
            (ep.name, ep.value, ep.load())
            for ep in importlib.metadata.entry_points(group=group)
        ]
    return [(ep.name, ep.value) for ep in importlib.metadata.entry_points(group=group)]


def load_plugin_class(group: str, name: str) -> type[abc.Plugin]:
    """Load a plugin class.

    Three formats are supported for the name of the plugin:

    1. The name of an installed plugin registered under the entry point group.

    2. A reference to an attribute name and the importable module it belongs to. The
       name "FromDisk:my_plugins.trajectories" effectively corresponds to this function
       being `from my_plugins.trajectories import FromDisk; return FromDisk`.

    3. The name of an attribute in a Python file and the path of the file. The name
       "FromDisk:/path/to/my_plugins.py" results in the function dynamically importing
       the `/path/to/my_plugins.py` file and returning the `FromDisk` attribute from it.
       Note that relative paths will be resolved relative to the current working
       directory.

    Parameters
    ----------
    group : str
        The name of the entry point group that the plugin belongs to, e.g.,
        "openstb.simulator.trajectory".
    name : str
        The user-specified name of the plugin.

    Returns
    -------
    plugin

    """
    classname, sep, modname_or_path = name.partition(":")

    # Separator not present: this is a registered plugin.
    if not sep:
        eps = importlib.metadata.entry_points().select(group=group, name=name)
        if not eps:
            raise ValueError(
                _("no {group} plugin named '{name}' is installed").format(
                    group=group, name=name
                )
            )
        if len(eps) > 1:
            raise ValueError(
                _("multiple {group} plugins named '{name}' are installed").format(
                    group=group, name=name
                )
            )
        return eps[name].load()

    # Not a registered plugin; does it refer to a file?
    path = Path(modname_or_path)
    if path.is_file():
        # Generate a unique module name. It is possible that two files at different
        # locations with the same stem could be requested; using just the stem could
        # result in the cached version of the first being used for the second.
        md5 = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
        modname = f"openstb.simulator.dynamic_plugins.{md5}.{path.stem}"

        # Generate an importlib spec for this module.
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            raise ValueError(
                _(
                    "could not create import specification for plugin file {path}"
                ).format(path=path)
            )

        # Create and load the module.
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # If the attribute is missing, raise a better error as the default will include
        # the hashed module name we generated.
        try:
            return getattr(mod, classname)
        except AttributeError:
            raise AttributeError(
                _("file {path} has no attribute {classname}").format(
                    path=path, classname=classname
                )
            )

    # Assume it is an installed module.
    mod = importlib.import_module(modname_or_path)
    return getattr(mod, classname)


class PluginSpecDict(TypedDict):
    """Specification for a plugin."""

    #: Name of the plugin. See `load_plugin_class` for the supported types.
    name: str

    #: Parameters for the plugin.
    parameters: dict[str, Any]


#: Specification for a plugin: either a `PluginSpecDict` or an existing instance of a
#: plugin.
PluginSpec = PluginSpecDict | abc.Plugin


def load_plugin(group: str, plugin_spec: PluginSpec) -> abc.Plugin:
    # Note: we cannot check isinstance(..., Mapping) here as plugins may implement the
    # Mapping interface.
    if isinstance(plugin_spec, dict):
        cls = load_plugin_class(group, plugin_spec["name"])
        return cls(**plugin_spec.get("parameters", {}))

    return plugin_spec


# Note: the cast() call in the following functions does no runtime checking, simply
# returning the value unchanged.  It is used to prevent static type checkers complaining
# about the return type.


def config_loader(plugin_spec: PluginSpec) -> abc.ConfigLoader:
    return cast(
        abc.ConfigLoader, load_plugin("openstb.simulator.config_loader", plugin_spec)
    )


def dask_cluster(plugin_spec: PluginSpec) -> abc.DaskCluster:
    return cast(
        abc.DaskCluster, load_plugin("openstb.simulator.dask_cluster", plugin_spec)
    )


def environment(plugin_spec: PluginSpec) -> abc.Environment:
    return cast(
        abc.Environment, load_plugin("openstb.simulator.environment", plugin_spec)
    )


def ping_times(plugin_spec: PluginSpec) -> abc.PingTimes:
    return cast(abc.PingTimes, load_plugin("openstb.simulator.ping_times", plugin_spec))


def point_targets(plugin_spec: PluginSpec) -> abc.PointTargets:
    return cast(
        abc.PointTargets, load_plugin("openstb.simulator.point_targets", plugin_spec)
    )


def result_converter(plugin_spec: PluginSpec) -> abc.ResultConverter:
    return cast(
        abc.ResultConverter,
        load_plugin("openstb.simulator.result_converter", plugin_spec),
    )


def scale_factor(plugin_spec: PluginSpec) -> abc.ScaleFactor:
    return cast(
        abc.ScaleFactor, load_plugin("openstb.simulator.scale_factor", plugin_spec)
    )


def signal(plugin_spec: PluginSpec) -> abc.Signal:
    return cast(abc.Signal, load_plugin("openstb.simulator.signal", plugin_spec))


def signal_window(plugin_spec: PluginSpec) -> abc.SignalWindow:
    """Load a signal window plugin.

    Parameters
    ----------
    plugin_spec : PluginSpec
        If a dictionary, this specifies the name and parameters of the signal window to
        load. Otherwise, it is assumed to be an instance of a compatible class and is
        returned unchanged.

    Returns
    -------
    signal_window : openstb.simulator.abc.SignalWindow

    """
    return cast(
        abc.SignalWindow, load_plugin("openstb.simulator.signal_window", plugin_spec)
    )


def system(plugin_spec: PluginSpec) -> abc.System:
    return cast(abc.System, load_plugin("openstb.simulator.system", plugin_spec))


def trajectory(plugin_spec: PluginSpec) -> abc.Trajectory:
    return cast(
        abc.Trajectory, load_plugin("openstb.simulator.trajectory", plugin_spec)
    )


def transducer(plugin_spec: PluginSpec) -> abc.Transducer:
    return cast(
        abc.Transducer, load_plugin("openstb.simulator.transducer", plugin_spec)
    )


def travel_time(plugin_spec: PluginSpec) -> abc.TravelTime:
    return cast(
        abc.TravelTime, load_plugin("openstb.simulator.travel_time", plugin_spec)
    )

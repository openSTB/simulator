# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Support for finding and loading plugins."""

from collections.abc import Mapping
import hashlib
import importlib.metadata
import importlib.util
import inspect
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    NotRequired,
    TypedDict,
    overload,
)

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc

_ = translations.load("openstb.simulator").gettext


class PluginSpec(TypedDict):
    """Specification for a plugin."""

    name: str
    """Name of the plugin.

    This can be in any format supported by the [load_plugin_class]
    [openstb.simulator.plugin.loader.load_plugin_class] function.

    """

    parameters: Mapping[str, Any]
    """Parameters for the plugin."""

    spec_source: NotRequired[str | PathLike[str]]
    """Source of this specification.

    For example the configuration file it was loaded from. This may be used for error
    reporting, and it may also be used by the plugin if it needs to load data based on
    the original source (e.g., from a file with a filename relative to the original
    configuration file).

    """


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
    group
        The name of the entry point group to list plugins for, e.g.,
        "openstb.simulator.trajectory".
    load
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


type PluginLoader[T] = Callable[[PluginSpec | T], T]
"""A callable which takes a plugin or specification and returns a plugin."""


def load_plugin_class[T](group: str, name: str) -> type[T]:
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
    group
        The name of the entry point group that the plugin belongs to, e.g.,
        "openstb.simulator.trajectory".
    name
        The user-specified name of the plugin.

    Returns
    -------
    plugin : class

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

    # Assume it is an installed module. If this intended as a filename but the file does
    # not exist, we will get either a ModuleNotFoundError or, if it starts with ., a
    # TypeError as a relative import requires package details.
    try:
        mod = importlib.import_module(modname_or_path)
    except (ModuleNotFoundError, TypeError):
        raise ValueError(
            _(
                "could not import plugin from {path}. Is the module not installed or "
                "the file missing?"
            ).format(path=path)
        )
    return getattr(mod, classname)


registered_loaders: dict[type, PluginLoader] = {}
"""Available plugin loader functions."""


def register_loader[T](loader: PluginLoader[T]) -> PluginLoader[T]:
    """Decorator to register a plugin loading function.

    This should be applied to a function which takes a
    `openstb.simulator.types.PluginOrSpec` parameter and returns a plugin instance.

    Parameters
    ----------
    loader
        The loader to register.

    Returns
    -------
    PluginLoader
        The original loader.

    """
    # Use the return annotation to do the registration.
    cls = inspect.signature(loader).return_annotation
    registered_loaders[cls] = loader

    return loader


def load_plugin[T](group: str, plugin_spec: PluginSpec | T) -> T:
    """Load a plugin and create an instance of it.

    In general, you should call one of the more specific functions in this module to
    load the particular type of plugin you want.

    Parameters
    ----------
    group
        The name of the entry point group that the plugin belongs to, e.g.,
        "openstb.simulator.trajectory".
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to
        load. Otherwise, it is assumed to be an instance of a compatible class and is
        returned unchanged.

    Returns
    -------
    instance : abc.Plugin
        If a plugin specification dictionary was given, an instance of the plugin it
        defined. Otherwise, the input object is returned unchanged.

    """
    # Note: we cannot check isinstance(..., Mapping) here as plugins may implement the
    # Mapping interface.
    if isinstance(plugin_spec, dict):
        cls: type[T] = load_plugin_class(group, plugin_spec["name"])

        # Start with parameters from the spec.
        params: dict[str, Any] = {}
        if "parameters" in plugin_spec:
            params.update(plugin_spec["parameters"])

        # If the plugin initialiser accepts a spec_source parameter, add that.
        cls_params = inspect.signature(cls).parameters
        if "spec_source" in cls_params and "spec_source" in plugin_spec:
            params["spec_source"] = plugin_spec["spec_source"]

        return cls(**params)

    return plugin_spec


@register_loader
def config_loader(plugin_spec: PluginSpec | abc.ConfigLoader) -> abc.ConfigLoader:
    """Load a configuration loader plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.ConfigLoader
        The configuration loader instance.

    """
    return load_plugin("openstb.simulator.config_loader", plugin_spec)


@register_loader
def controller(plugin_spec: PluginSpec | abc.Controller) -> abc.Controller:
    """Load a simulation controller plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Controller
        The simulation controller instance.

    """
    return load_plugin("openstb.simulator.controller", plugin_spec)


@register_loader
def dask_cluster(plugin_spec: PluginSpec | abc.DaskCluster) -> abc.DaskCluster:
    """Load a Dask cluster plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.DaskCluster
        The Dask cluster instance.

    """
    return load_plugin("openstb.simulator.dask_cluster", plugin_spec)


@register_loader
def distortion(plugin_spec: PluginSpec | abc.Distortion) -> abc.Distortion:
    """Load a distortion plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Distortion
        The distortion instance.

    """
    return load_plugin("openstb.simulator.distortion", plugin_spec)


@register_loader
def environment(plugin_spec: PluginSpec | abc.Environment) -> abc.Environment:
    """Load a environment plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Environment
        The environment instance.

    """
    return load_plugin("openstb.simulator.environment", plugin_spec)


@register_loader
def ping_times(plugin_spec: PluginSpec | abc.PingTimes) -> abc.PingTimes:
    """Load a ping time plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.PingTimes
        The ping times instance.

    """
    return load_plugin("openstb.simulator.ping_times", plugin_spec)


@register_loader
def point_targets(plugin_spec: PluginSpec | abc.PointTargets) -> abc.PointTargets:
    """Load a point targets plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.PointTargets
        The point targets instance.

    """
    return load_plugin("openstb.simulator.point_targets", plugin_spec)


@register_loader
def result_converter(
    plugin_spec: PluginSpec | abc.ResultConverter,
) -> abc.ResultConverter:
    """Load a result converter plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.ResultConverter
        The result converter instance.

    """
    return load_plugin("openstb.simulator.result_converter", plugin_spec)


@register_loader
def scattering_model(
    plugin_spec: PluginSpec | abc.ScatteringModel,
) -> abc.ScatteringModel:
    """Load a scattering model plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.ScatteringModel
        The scattering model instance.

    """
    return load_plugin("openstb.simulator.scattering_model", plugin_spec)


@register_loader
def signal(plugin_spec: PluginSpec | abc.Signal) -> abc.Signal:
    """Load a signal plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Signal
        The signal instance.

    """
    return load_plugin("openstb.simulator.signal", plugin_spec)


@register_loader
def signal_window(plugin_spec: PluginSpec | abc.SignalWindow) -> abc.SignalWindow:
    """Load a signal window plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.SignalWindow
        The signal window instance.

    """
    return load_plugin("openstb.simulator.signal_window", plugin_spec)


@register_loader
def system(plugin_spec: PluginSpec | abc.System) -> abc.System:
    """Load a system plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.System
        The system instance.

    """
    return load_plugin("openstb.simulator.system", plugin_spec)


@register_loader
def trajectory(plugin_spec: PluginSpec | abc.Trajectory) -> abc.Trajectory:
    """Load a trajectory plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Trajectory
        The trajectory instance.

    """
    return load_plugin("openstb.simulator.trajectory", plugin_spec)


@register_loader
def transducer(plugin_spec: PluginSpec | abc.Transducer) -> abc.Transducer:
    """Load a transducer plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.Transducer
        The transducer instance.

    """
    return load_plugin("openstb.simulator.transducer", plugin_spec)


@register_loader
def travel_time(plugin_spec: PluginSpec | abc.TravelTime) -> abc.TravelTime:
    """Load a travel time plugin.

    Parameters
    ----------
    plugin_spec
        If a dictionary, this specifies the name and parameters of the plugin to load.
        Otherwise, it is assumed to be an instance of a compatible class and is returned
        unchanged.

    Returns
    -------
    abc.TravelTime
        The travel time instance.

    """
    return load_plugin("openstb.simulator.travel_time", plugin_spec)

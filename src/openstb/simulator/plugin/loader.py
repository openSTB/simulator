# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Literal, overload

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc
from openstb.simulator.types import F_PluginLoader, PluginOrSpec, T_Plugin

_ = translations.load("openstb.simulator").gettext


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


def load_plugin_class(group: str, name: str) -> type[T_Plugin]:
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

    # Assume it is an installed module.
    mod = importlib.import_module(modname_or_path)
    return getattr(mod, classname)


#: Lookup table of functions which have been registered as able to load a particular
#: class of plugin.
registered_loaders: dict[type, F_PluginLoader] = {}


# For when the decorator is used directly (without parameters).
@overload
def register_loader(__loader: F_PluginLoader[T_Plugin]) -> F_PluginLoader[T_Plugin]: ...


# For when the decorator is used with parameters.
@overload
def register_loader(
    *, docstring: bool | None = None, name: str | None = None
) -> Callable[[F_PluginLoader[T_Plugin]], F_PluginLoader[T_Plugin]]: ...


def register_loader(
    __loader: F_PluginLoader[T_Plugin] | None = None,
    *,
    docstring: bool | None = None,
    name: str | None = None,
):
    """Decorator to register a plugin loading function.

    This should be applied to a function which takes a
    `openstb.simulator.types.PluginOrSpec` parameter and returns a plugin instance.

    Parameters
    ----------
    docstring : Boolean, optional
        Whether to set a docstring for the class. If None (the default), then a
        docstring will only be set if one does not already exist.
    name : str, optional
        The name of the plugin type to use in the docstring. If not specified, the
        class name will be used.

    """

    def registration(loader: F_PluginLoader[T_Plugin]) -> F_PluginLoader[T_Plugin]:
        # Use the return annotation to do the actual registration.
        cls = inspect.signature(loader).return_annotation
        registered_loaders[cls] = loader

        # Check if we need to set a doctring.
        if docstring is None:
            set_docstring = loader.__doc__ is None or len(loader.__doc__.strip()) == 0
        else:
            set_docstring = docstring
        if set_docstring:
            docname = name or cls.__name__
            loader.__doc__ = f"""Load a {docname} plugin.

Parameters
----------
plugin_spec : dict, {cls.__name__}
    If a dictionary, this specifies the name and parameters of the {docname} to
    load. Otherwise, it is assumed to be an instance of a compatible class and is
    returned unchanged.

Returns
-------
{cls.__name__}"""

        return loader

    # Directly used.
    if __loader is not None:
        return registration(__loader)

    # Used with parameters.
    return registration


def load_plugin(group: str, plugin_spec: PluginOrSpec[T_Plugin]) -> T_Plugin:
    """Load a plugin and create an instance of it.

    In general, you should call one of the more specific functions in this module to
    load the particular type of plugin you want.

    Parameters
    ----------
    group : str
        The name of the entry point group that the plugin belongs to, e.g.,
        "openstb.simulator.trajectory".
    plugin_spec : dict, Plugin
        If a dictionary, this specifies the name and parameters of the plugin to
        load. Otherwise, it is assumed to be an instance of a compatible class and is
        returned unchanged.

    Returns
    -------
    abc.Plugin

    """
    # Note: we cannot check isinstance(..., Mapping) here as plugins may implement the
    # Mapping interface.
    if isinstance(plugin_spec, dict):
        cls: type[T_Plugin] = load_plugin_class(group, plugin_spec["name"])

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


@register_loader(name="configuration loader")
def config_loader(plugin_spec: PluginOrSpec[abc.ConfigLoader]) -> abc.ConfigLoader:
    return load_plugin("openstb.simulator.config_loader", plugin_spec)


@register_loader(name="Dask cluster")
def dask_cluster(plugin_spec: PluginOrSpec[abc.DaskCluster]) -> abc.DaskCluster:
    return load_plugin("openstb.simulator.dask_cluster", plugin_spec)


@register_loader(name="echo signal distortion")
def distortion(plugin_spec: PluginOrSpec[abc.Distortion]) -> abc.Distortion:
    return load_plugin("openstb.simulator.distortion", plugin_spec)


@register_loader(name="environmental parameters")
def environment(plugin_spec: PluginOrSpec[abc.Environment]) -> abc.Environment:
    return load_plugin("openstb.simulator.environment", plugin_spec)


@register_loader(name="ping times")
def ping_times(plugin_spec: PluginOrSpec[abc.PingTimes]) -> abc.PingTimes:
    return load_plugin("openstb.simulator.ping_times", plugin_spec)


@register_loader(name="point targets")
def point_targets(plugin_spec: PluginOrSpec[abc.PointTargets]) -> abc.PointTargets:
    return load_plugin("openstb.simulator.point_targets", plugin_spec)


@register_loader(name="result converter")
def result_converter(spec: PluginOrSpec[abc.ResultConverter]) -> abc.ResultConverter:
    return load_plugin("openstb.simulator.result_converter", spec)


@register_loader(name="signal")
def signal(plugin_spec: PluginOrSpec[abc.Signal]) -> abc.Signal:
    return load_plugin("openstb.simulator.signal", plugin_spec)


@register_loader(name="signal window")
def signal_window(plugin_spec: PluginOrSpec[abc.SignalWindow]) -> abc.SignalWindow:
    return load_plugin("openstb.simulator.signal_window", plugin_spec)


@register_loader(name="simulation")
def simulation(plugin_spec: PluginOrSpec[abc.Simulation]) -> abc.Simulation:
    return load_plugin("openstb.simulator.simulation", plugin_spec)


@register_loader(name="system")
def system(plugin_spec: PluginOrSpec[abc.System]) -> abc.System:
    return load_plugin("openstb.simulator.system", plugin_spec)


@register_loader(name="trajectory")
def trajectory(plugin_spec: PluginOrSpec[abc.Trajectory]) -> abc.Trajectory:
    return load_plugin("openstb.simulator.trajectory", plugin_spec)


@register_loader(name="transducer")
def transducer(plugin_spec: PluginOrSpec[abc.Transducer]) -> abc.Transducer:
    return load_plugin("openstb.simulator.transducer", plugin_spec)


@register_loader(name="travel time")
def travel_time(plugin_spec: PluginOrSpec[abc.TravelTime]) -> abc.TravelTime:
    return load_plugin("openstb.simulator.travel_time", plugin_spec)

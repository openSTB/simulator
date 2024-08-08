# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Mapping
from os import PathLike
from typing import Any, Callable, NotRequired, TypedDict, TypeVar, Union


class PluginSpec(TypedDict):
    """Specification for a plugin."""

    #: Name of the plugin. See `load_plugin_class` for the supported types.
    name: str

    #: Parameters for the plugin.
    parameters: Mapping[str, Any]

    #: Source of this specification, for example the configuration file it was loaded
    #: from. This may be used for error reporting, and it may also be used by the plugin
    #: if it needs to load data based on the original source (e.g., from a file with a
    #: filename relative to the original configuration file).
    spec_source: NotRequired[str | PathLike[str]]


#: Generic type variable representing some type of plugin.
T_Plugin = TypeVar("T_Plugin")


#: Either a plugin, or a plugin specification dictionary.
#: This can be specialised, e.g., PluginOrSpec[abc.Signal].
PluginOrSpec = Union[T_Plugin, PluginSpec]


#: A callable which takes a plugin or specification and returns a plugin.
F_PluginLoader = Callable[[PluginOrSpec[T_Plugin]], T_Plugin]


#: Generic type for the configuration of a simulation. Each type of simulation will
#: require a different configuration structure. Here we simply say it will be a mapping
#: from string keys to any type of object, and allow the plugins to define a more
#: specific type.
SimulationConfig = TypeVar("SimulationConfig", bound=Mapping[str, Any])

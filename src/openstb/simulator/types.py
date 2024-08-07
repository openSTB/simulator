# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Mapping
from typing import Any, TypedDict, TypeVar, Union

from openstb.simulator.plugin.abc import Plugin


class PluginSpec(TypedDict):
    """Specification for a plugin."""

    #: Name of the plugin. See `load_plugin_class` for the supported types.
    name: str

    #: Parameters for the plugin.
    parameters: Mapping[str, Any]


#: Generic type variable representing some type of plugin.
T_Plugin = TypeVar("T_Plugin", bound=Plugin)


#: Either a plugin, or a plugin specification dictionary.
#: This can be specialised, e.g., PluginOrSpec[abc.Signal].
PluginOrSpec = Union[T_Plugin, PluginSpec]

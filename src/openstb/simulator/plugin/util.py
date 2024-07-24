# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import MutableMapping
from typing import Any

from openstb.i18n.support import domain_translator
from openstb.simulator import abc


_ = domain_translator("openstb.simulator")


def flatten_system(
    config: MutableMapping[str, Any],
    system: str = "system",
    transmitter: str | None = "transmitter",
    receivers: str | None = "receivers",
    signal: str | None = "signal",
):
    """Extract system-defined plugins and place at the top level of the configuration.

    The system plugin is a convenience plugin to group system-related properties. This
    function is intended for simulation methods which want to support either a system
    plugin or individually-specified plugins. It extracts the plugins provided by the
    system plugin and adds them to the top level of the configuration. If a plugin is
    specified both by the system and directly in the configuration, an error is raised.

    Parameters
    ----------
    config : mapping
        The configuration to modify.
    system : str
        The key that the system plugin would be stored under.
    transmitter, receivers, signal : str, None
        The keys to store the system-provided plugins under. Set to None to ignore that
        type of plugin.

    """
    system_plugin: abc.System = config.get(system, None)
    if system_plugin is None:
        return

    if transmitter is not None:
        system_tx = system_plugin.transmitter
        if system_tx is not None:
            config_tx = config.get(transmitter, None)
            if config_tx is not None:
                raise ValueError(
                    _(
                        "the system plugin and the simulation configuration both "
                        "specify a transmitter"
                    )
                )
            config[transmitter] = system_tx

    if receivers is not None:
        system_rx = system_plugin.receivers
        if system_rx is not None:
            config_rx = config.get(receivers, None)
            if config_rx is not None:
                raise ValueError(
                    _(
                        "the system plugin and the simulation configuration both "
                        "specify receivers"
                    )
                )
            config[receivers] = system_rx

    if signal is not None:
        system_signal = system_plugin.signal
        if system_signal is not None:
            config_signal = config.get(signal, None)
            if config_signal is not None:
                raise ValueError(
                    _(
                        "the system plugin and the simulation configuration both "
                        "specify a signal"
                    )
                )
            config[signal] = system_signal

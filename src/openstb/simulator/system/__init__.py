# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from openstb.simulator.plugin import loader
from openstb.simulator.plugin.abc import Signal, System, Transducer
from openstb.simulator.types import PluginOrSpec


class GenericSystem(System):
    """A generic system grouping other plugins."""

    def __init__(
        self,
        transmitter: PluginOrSpec[Transducer] | None,
        receivers: list[PluginOrSpec[Transducer]] | None,
        signal: PluginOrSpec[Signal] | None,
    ):
        """
        Parameters
        ----------
        transmitter : PluginSpec, optional
            The transducer to use for transmission.
        receivers : list[PluginSpec], optional
            The transducers to be used as receivers.
        signal : PluginSpec, optional
            The signal transmitted by the system.

        """
        self._transmitter = None
        if transmitter is not None:
            self._transmitter = loader.transducer(transmitter)

        self._receivers = None
        if receivers is not None:
            self._receivers = [loader.transducer(rx) for rx in receivers]

        self._signal = None
        if signal is not None:
            self._signal = loader.signal(signal)

    @property
    def transmitter(self) -> Transducer | None:
        return self._transmitter

    @property
    def receivers(self) -> list[Transducer] | None:
        return self._receivers

    @property
    def signal(self) -> Signal | None:
        return self._signal

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import Sequence

from openstb.simulator.plugin import loader
from openstb.simulator.plugin.abc import Signal, System, Transducer
from openstb.simulator.types import PluginOrSpec


class GenericSystem(System):
    """A generic system grouping other plugins."""

    def __init__(
        self,
        transmitter: PluginOrSpec[Transducer],
        receivers: Sequence[PluginOrSpec[Transducer]],
        signal: PluginOrSpec[Signal],
    ):
        """
        Parameters
        ----------
        transmitter : PluginSpec
            The transducer to use for transmission.
        receivers : list[PluginSpec]
            The transducers to be used as receivers.
        signal : PluginSpec
            The signal transmitted by the system.

        """
        self._transmitter = loader.transducer(transmitter)
        self._receivers = [loader.transducer(rx) for rx in receivers]
        self._signal = loader.signal(signal)

    @property
    def transmitter(self) -> Transducer:
        return self._transmitter

    @property
    def receivers(self) -> list[Transducer]:
        return self._receivers

    @property
    def signal(self) -> Signal:
        return self._signal

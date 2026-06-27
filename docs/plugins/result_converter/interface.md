---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Result converter plugin interface

The base class defining the interface expected of a result converter plugin is
[plugin.abc.ResultConverter][openstb.simulator.plugin.abc.ResultConverter]. Plugins are
registered under the `openstb.simulator.result_converter` entry point.


## Support check

The plugin must provide a [`can_handle`][openstb.simulator.plugin.abc.ResultConverter.can_handle]
method which a simulation controller can use to check its result format is supported
prior to beginning the simulation. This will be given the result format and the
configuration dictionary for the simulation, and must return a Boolean value indicating
whether it will be able to convert these results. The result format may be given as
either a value from the [ResultFormat][openstb.simulator.plugin.abc.ResultFormat]
enumeration, or a string to refer to a custom format used by an external simulation
controller.


## Conversion

The plugin must provide a [`convert`][openstb.simulator.plugin.abc.ResultConverter.convert]
method to perform the conversion. This will be given the result format (see the support
check section above for details on possible values), the simulation results and the
configuration dictionary for the simulation. It must return a Boolean value indicating
if the conversion was successful. If successful, the simulation controller may delete
the original results. If unsuccessful, the simulation controller should retain the
original results.


## Example

The following plugin would save only the recorded pressures of a [baseband Zarr][openstb.simulator.plugin.abc.ResultFormat.ZARR_BASEBAND_PRESSURE]
result to a NumPy file.

```python
from typing import Any

import numpy as np

from openstb.simulator.plugin import abc


class PressureNPYConverter(abc.ResultConverter):
    """Save just the pressure as a .npy file."""

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename
            Filename to save the results under.

        """
        self.filename = filename

    def can_handle(
        self,
        format: abc.ResultFormat | str,
        config: abc.ControllerConfig,
    ) -> bool:
        # This is the only format we support converting.
        return format == abc.ResultFormat.ZARR_BASEBAND_PRESSURE

    def convert(
        self,
        format: abc.ResultFormat | str,
        result: Any,
        config: abc.ControllerConfig,
    ) -> bool:
        # Sanity check.
        if format != abc.ResultFormat.ZARR_BASEBAND_PRESSURE:
            return False

        # Save the pressure.
        np.save(self.filename, result["pressure"][:])

        # Original results may now be deleted.
        return True
```

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
from typing import Any, Literal

import scipy.io

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import ResultConverter, ResultFormat, SimulationConfig

_ = translations.load("openstb.simulator").gettext


class MATLABConverter(ResultConverter):
    """Convert simulation results to a MATLAB file.

    This writes the data with the following variables:

    * ``baseband_frequency``: the frequency used to baseband the data
    * ``ping_start_time``: seconds since the start of the trajectory that the pings
      were sent
    * ``pressure``: the simulated pressure at each receiver
    * ``pressure_dimensions``: string array giving the order of dimensions in
      ``pressure``.
    * ``sample_time``: seconds since the start of its ping that each sample was captured

    Parameters
    ----------
    filename : path-like
        The path to save the converted results at.
    format, long_field_name, do_compression, oned_as
        Parameters passed to `scipy.io.savemat` when writing the data; see its docstring
        for details.

    """

    def __init__(
        self,
        filename: os.PathLike[str] | str,
        format: Literal["5"] | Literal["4"] = "5",
        long_field_names: bool = False,
        do_compression: bool = False,
        oned_as: Literal["row"] | Literal["column"] = "row",
    ):
        self.filename = Path(filename)
        self.format = format
        self.long_field_names = long_field_names
        self.do_compression = do_compression
        self.oned_as = oned_as

        if self.filename.exists():
            raise ValueError(
                _("output file {filename} already exists").format(
                    filename=self.filename
                )
            )

    def can_handle(self, format: ResultFormat | str, config: SimulationConfig):
        return format == ResultFormat.ZARR_BASEBAND_PRESSURE

    def convert(
        self, format: ResultFormat | str, result: Any, config: SimulationConfig
    ):
        if format != ResultFormat.ZARR_BASEBAND_PRESSURE:
            return False

        mdict = {
            "ping_start_time": result["ping_start_time"][:],
            "sample_time": result["sample_time"][:],
            "pressure": result["pressure"][:],
            "pressure_dimensions": ["ping", "channel", "sample_time"],
            "baseband_frequency": result.attrs["baseband_frequency"],
        }

        scipy.io.savemat(
            self.filename,
            mdict,
            appendmat=False,
            format=self.format,
            long_field_names=self.long_field_names,
            do_compression=self.do_compression,
            oned_as=self.oned_as,
        )

        return True

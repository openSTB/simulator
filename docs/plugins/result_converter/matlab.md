---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# MATLAB result converter

**Plugin name**: `matlab`
<br>
**Implementation**: [`openstb.simulator.result_converter.matlab.MATLABConverter`][]

The MATLAB result converter converts simulation results into the [.mat binary format][.mat]
used by [MATLAB][]. Note that it only supports format versions earlier than 7.3. The
file is written using the [`savemat`][scipy.io.savemat] function provided by SciPy; any
limitations of that function also apply to this converter.

[.mat]: https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html
[MATLAB]: https://www.mathworks.com/help/matlab/


## Supported result formats

The MATLAB converter supports converting results in the [Zarr baseband
pressure](../../concepts/result_format.md#zarr-baseband-pressure) format.


## Saved variables

The following variables are written into the MATLAB file:

* `baseband_frequency`: the frequency used to baseband the data
* `ping_start_time`: seconds since the start of the trajectory that the pings were sent
* `pressure`: the simulated pressure at each receiver
* `pressure_dimensions`: string array giving the order of dimensions in `pressure`.
* `sample_time`: seconds since the start of its ping that each sample was captured


## Parameters

The only required parameter for the plugin is the filename to save the result at:

```toml
[result_converter]
plugin = "matlab"
filename = "sim_results.mat"
```

Several other parameters are available, based upon options supported by the
[`savemat`][scipy.io.savemat] function:

* `format`: the MATLAB format version to use.
* `long_field_names`: whether to enable long field names.
* `do_compression`: whether to compress the variables within the file.
* `oned_as`: whether to save one-dimensional arrays as row vectors or column vectors.

The following example configuration shows the default values for these parameters. See
the documentation of the [`savemat`][scipy.io.savemat] function for further details of
the parameters including the allowed values.

```toml
[result_converter]
plugin = "matlab"
filename = "sim_results.mat"
format = "5"
long_field_names = false
do_compression = false
oned_as = "row"
```

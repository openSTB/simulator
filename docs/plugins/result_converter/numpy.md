---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# NumPy result converter

**Plugin name**: `numpy`
<br>
**Implementation**: [`openstb.simulator.result_converter.numpy.NumpyConverter`][]

The NumPy result converter saves simulation results as NumPy NPZ files. These are zipped
archives of one or more arrays which may be subsequently loaded with the [numpy.load][]
function.


## Supported result formats

The NumPy converter supports converting results in the [Zarr baseband
pressure](../../concepts/result_format.md#zarr-baseband-pressure) format.


## Saved variables

The following variables are written into the NPZ file:

* `baseband_frequency`: the frequency used to baseband the data
* `ping_start_time`: seconds since the start of the trajectory that the pings were sent
* `pressure`: the simulated pressure at each receiver
* `pressure_dimensions`: string array giving the order of dimensions in `pressure`.
* `sample_time`: seconds since the start of its ping that each sample was captured


## Parameters

The only required parameter for the converter is the filename to save the results as:

```toml
[result_converter]
plugin = "numpy"
filename = "sim_results.npz"
```

By default, the file is not compressed. To enable compression, use the `compress`
parameter:

```toml
[result_converter]
plugin = "numpy"
filename = "sim_results.npz"
compress = true
```

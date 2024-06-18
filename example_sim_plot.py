# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from matplotlib import pyplot as plt
import numpy as np
import zarr


# The output is stored in a zarr file. This is essentially a directory with
# sub-directories for nested groups and files for each chunk of an array. It is designed
# to be used in parallel environments.
results = zarr.open("example_sim.zarr")

# The arrays within the zarr file can be loaded as NumPy arrays.
t = results["sample_time"][:]
raw = results["results"][:]

# As an example, plot the trace from the middle receiver at ping 14 as sound pressure
# level (dB relative to 1uPa). The results have the dimensions (ping, receiver, time).
trace = raw[14, 2, :]
plt.figure()
plt.plot(t, 20 * np.log10(np.abs(trace) / 1e-6))
plt.xlabel("Time (s)")
plt.ylabel("Echo strength (SPL)")
plt.show()

# Or an image of all pings recorded on the first receiver.
rx = raw[:, 0, :]
plt.figure()
plt.imshow(
    20 * np.log10(np.abs(rx) / 1e-6),
    aspect="auto",
    origin="lower",
    interpolation="none",
    extent=(t[0], t[-1], 0, raw.shape[0] - 1),
    vmin=100,
)
plt.colorbar(label="Echo strength (SPL)")
plt.xlabel("Time (s)")
plt.ylabel("Ping")
plt.title("Receiver 0")
plt.show()

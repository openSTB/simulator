# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from matplotlib import pyplot as plt
import numpy as np

# This assumes you used the NumPy output converter in example_sim.py.
results = np.load("simple_points.npz")

# This loads the results as a mapping instance.
t = results["sample_time"]
P = results["pressure"]

# As an example, plot the trace from the middle receiver at ping 14 as sound pressure
# level (dB relative to 1uPa). The results have the dimensions ping, receiver, time
# (which is stored in a pressure_dimensions array in the results).
trace = P[14, 2, :]
plt.figure()
plt.plot(t, 20 * np.log10(np.abs(trace) / 1e-6))
plt.title("Middle receiver, ping 14")
plt.xlabel("Time (s)")
plt.ylabel("Echo strength (SPL)")
plt.show()

# Or an image of all pings recorded on the first receiver.
rx = P[:, 0, :]
plt.figure()
plt.imshow(
    20 * np.log10(np.abs(rx) / 1e-6),
    aspect="auto",
    origin="lower",
    interpolation="none",
    extent=(t[0], t[-1], 0, P.shape[0] - 1),
    vmin=100,
)
plt.colorbar(label="Echo strength (SPL)")
plt.xlabel("Time (s)")
plt.ylabel("Ping")
plt.title("Receiver 0")
plt.show()

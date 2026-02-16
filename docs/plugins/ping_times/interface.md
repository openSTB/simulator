---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Plugin interface

The base class defining the interface expected of a ping time calculator is
[plugin.abc.PingTimes][openstb.simulator.plugin.abc.PingTimes]. Plugins are registered
under the  `openstb.simulator.ping_times` entry point.


## Calculate method

A plugin must provide a `calculate` method which is given the trajectory followed by the
system during the simulation. This method must return a one-dimensional floating-point
array giving the ping times in seconds since the start of the trajectory.


## Example

The following plugin would determine the ideal ping times to have a constant interval
between pings, and then add some jitter with a Gaussian distribution.

```python
import numpy as np

from openstb.simulator.plugin import abc


class JitterPingTimes(abc.PingTimes):
    """Ping times with some jitter around a mean interval."""

    def __init__(self, seed: int, mean_interval: float, jitter_std: float):
        """
        Parameters
        ----------
        seed
            Seed for the random number generator creating the jitter.
        mean_interval
            Desired mean interval between pings in seconds.
        jitter_std
            Standard deviation of the jitter in seconds.

        """
        self.rng = np.random.default_rng(seed)
        self.mean_interval = mean_interval
        self.jitter_std = jitter_std

    def calculate(self, trajectory: abc.Trajectory) -> np.ndarray:
        mean_times = np.arange(0, trajectory.duration, self.mean_interval)
        return mean_times + self.rng.normal(0, self.jitter_std, len(mean_times))
```

To register this plugin, add an entry point of the following format to the standard
[pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
file of your project.

```toml
[project.entry-points."openstb.simulator.ping_times"]
jitter = "my_package.ping_time:JitterPingTimes"
```

---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# Specifying plugins

## Registered plugins

All plugins included with the simulator are *registered* on installation. Most
third-party plugins will also be registered when you install the corresponding package.
This registration utilises the  [entry point mechanism][ep] provided by the Python
packaging format which maps a name to a particular piece of code. For plugins, this
means you can simply use the registered name to specify it. For example, the linear
trajectory plugin is registered under the name `linear` meaning that you can refer to it
with this name:

```toml
[trajectory]
plugin = "linear"
start_position = [0, 0, 0]
end_position = [100, 0, 0]
speed = 1.75
```

Note that each type of plugin (trajectories, targets, systems etc) has an independent
set of registered plugins. This means it is possible for two plugins in different
categories to have the same registered name.

[ep]: https://packaging.python.org/en/latest/specifications/entry-points/


## Unregistered plugins

Plugins can be loaded directly from any Python module. This allows you to specify a
plugin which has not been registered. This may be useful for testing during the
development of a plugin, but in general it is recommended that plugins are registered
for ease of use. For an unregistered plugin, instead of specifying a name you need to
specify the name of the class implementing the plugin and the module to load it from in
the format `Class:package.module.submodule`.

For example, the linear trajectory plugin is provided by the
[Linear][openstb.simulator.system.trajectory.Linear] class in the
[openstb.simulator.system.trajectory][] module. The same trajectory configuration as
shown in the previous section can be achieved by loading it as follows:

```toml
[trajectory]
plugin = "Linear:openstb.simulator.system.trajectory"
start_position = [0, 0, 0]
end_position = [100, 0, 0]
speed = 1.75
```

## Directly from a file

It is also possible to load a plugin directly from a Python file which
is not part of an installed package. This can be useful during plugin development, and
may also be useful for plugins which are highly specific to a particular experiment
where it makes more sense to keep the plugin with the simulation setup instead of in a
separate package.

In this case, the plugin must be specified in the format `Class:/path/to/myfile.py`.
Relative paths (such as `Class:myfile.py`) will be resolved from the current working
directory when the plugin is being loaded. In the following example, a custom `Arc`
trajectory plugin is loaded from the `my_trajectories.py` file in the same directory as
the configuration file.

=== "Configuration"

    ```toml
    [trajectory]
    plugin = "Arc:my_trajectories.py"
    start_angle = 0
    end_angle = 90
    radius = 50
    speed = 1.2
    ```
=== "my_trajectories.py"

    ```python
    class Arc:
        def __init__(
            self, start_angle: float, end_angle: float, radius: float, speed: float
        ):
            # Store the parameters.
            self.start_angle = start_angle
            self.end_angle = end_angle
            self.radius = radius
            self.speed = speed

        # Actual implementation left to the reader's imagination.
    ```


## Plugin validity

The code that loads the plugins does not attempt to check that the plugin you have
specified has the expected interface. The simulator will assume that, if you have
specified a particular plugin to define (for example) the trajectory then you know it is
suitable for this purpose. If it is not, it may cause an exception during the
initialisation of the simulation or during the actual simulation process.

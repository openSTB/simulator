# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import click

from openstb.simulator.plugin import loader, util


@click.command
@click.argument("config_source", nargs=1, type=str)
@click.option(
    "-c",
    "--config-plugin",
    type=str,
    default=None,
    metavar="PLUGIN",
    help=(
        "The name of the configuration loading plugin needed to parse the "
        "configuration source. This can be the name of a registered plugin, a "
        "'ClassName:package.module' reference to a class in an installed module, or a "
        "'ClassName:path/to/file.py' reference to a class in a Python file."
    ),
)
def run(config_plugin, config_source):
    """Run a simulation.

    The configuration of the simulation is loaded from the source given by
    CONFIG_SOURCE. This may be the path to a configuration file, or another type of
    source handled by a suitable configuration loading plugin.

    """
    loader_cls = None

    # Loader plugin specified; get it.
    if config_plugin is not None:
        loader_cls = loader.load_plugin_class(
            "openstb.simulator.config_loader", config_plugin
        )

    # Not specified; try to guess it.
    else:
        name, loader_cls = util.find_config_loader(config_source)
        if loader_cls is None:
            raise click.ClickException(
                "could not automatically determine the config loader; please specify "
                "the appropriate loader."
            )

    # Create an instance and get the configuration dictionary.
    config_loader = loader_cls(config_source)
    config = config_loader.load()

    # Get the simulation plugin.
    simulation = loader.simulation(config.pop("simulation"))

    # Load all the other plugins, and check it conforms to the configuration required by
    # the simulation plugin.
    util.load_config_plugins(simulation.config_class, config)

    # And then we can run the simulation.
    simulation.run(config)

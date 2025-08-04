# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import logging

import click
from rich.logging import RichHandler

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
@click.option(
    "--dask-worker",
    type=str,
    default=None,
    metavar="PLUGIN",
    help=(
        "For use with Dask cluster plugins which support independently starting the "
        "workers, such as clusters using MPI. This allows the workers to wait for "
        "configuration details from the main process instead of each worker parsing "
        "the configuration. Not all Dask cluster plugins will support this. The value "
        "can be the name of a registered plugin, a 'ClassName:package.module' "
        "reference to a class in an installed module, or a 'ClassName:path/to/file.py' "
        "reference to a class in a Python file."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"], case_sensitive=False
    ),
    default="info",
    # metavar="LEVEL",
    help=(
        "Set the level of log messages to display. Log messages with a lower level "
        "will be discarded."
    ),
)
def run(config_plugin, config_source, dask_worker, log_level):
    """Run a simulation.

    The configuration of the simulation is loaded from the source given by
    CONFIG_SOURCE. This may be the path to a configuration file, or another type of
    source handled by a suitable configuration loading plugin.

    """
    # Configure the display of log messages.
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    loader_cls = None

    # Given a Dask cluster plugin which supports independent worker spawning.
    if dask_worker is not None:
        cls = loader.load_plugin_class("openstb.simulator.dask_cluster", dask_worker)
        if not cls.initialise_worker():
            return

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

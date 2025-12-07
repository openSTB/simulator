# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import click

from openstb.simulator.cli.run import run
from openstb.simulator.cli.seed import seed


@click.group(
    context_settings=dict(help_option_names=("-h", "--help"), show_default=True)
)
def openstb_sim():
    """The openSTB sonar simulator."""
    pass


openstb_sim.add_command(run)
openstb_sim.add_command(seed)

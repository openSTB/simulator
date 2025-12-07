# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent


import click

from openstb.simulator.random import generate_seed


@click.command
@click.option(
    "--number", "-n", default=1, metavar="N", help="The number of seeds to generate."
)
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    show_default=False,
    help="Where to write the seeds. By default they are printed to the terminal.",
)
@click.option(
    "--digits",
    "-d",
    type=int,
    default=None,
    show_default=False,
    help=(
        "The maximum number of digits to output. By default all generated digits are "
        "output."
    ),
)
def seed(number, output, digits):
    """Generate random seeds.

    This uses NumPy's random number support to create seeds from available sources of
    entropy. They are not guaranteed to be cryptographically secure, but are suitable
    for seeding simulation plugins that need random numbers.

    """
    if number < 1:
        raise click.ClickException("number of seeds cannot be less than one")
    if digits is not None and digits < 1:
        raise click.ClickException("number of digits cannot be less than one")

    for i in range(number):
        seedstr = str(generate_seed())
        if digits is not None:
            seedstr = seedstr[:digits]
        output.write(seedstr)
        output.write("\n")

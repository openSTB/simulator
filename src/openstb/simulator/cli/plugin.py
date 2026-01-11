# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import importlib.metadata
from io import TextIOWrapper
import re

import click

from openstb.simulator.plugin import abc


@click.group
def plugin():
    """Find registered plugins."""
    pass


@plugin.command("list")
@click.argument("plugin-type", default=None, metavar="TYPE")
@click.option(
    "--output",
    "-o",
    type=click.File(mode="w"),
    default="-",
    show_default=False,
    help=(
        "Where to write the list of plugins. By default the list is written to the "
        "terminal."
    ),
)
def list_plugins(plugin_type: str | None, output: TextIOWrapper):
    """List registered plugins.

    This lists the registered simulation plugins of the given type. The type can be the
    full plugin entry point (e.g., openstb.simulator.distortion) or for convencience
    just the final name (e.g., distortion). If no type is specified all registered
    plugins are listed.

    """
    if plugin_type is None:
        for group in sorted(importlib.metadata.entry_points().groups):
            if not group.startswith("openstb.simulator"):
                continue
            _output_group(group, output)
    else:
        if "." not in plugin_type:
            plugin_type = f"openstb.simulator.{plugin_type}"
        _output_group(plugin_type, output)


def _output_group(group: str, output: TextIOWrapper):
    """List a group of registered plugins.

    Parameters
    ----------
    group
        The full group name, i.e., "openstb.simulator.distortion" not "distortion".
    output
        The stream to write the output to.

    """
    click.secho(f"\n{group}", bold=True, file=output)
    click.echo(f"{'-' * len(group)}", file=output)

    eps = importlib.metadata.entry_points(group=group)
    if not eps:
        output.write("No registered plugins\n")
        return

    for ep in eps:
        cls = ep.load()

        if ep.dist is not None:
            if ep.dist.name == "openstb-simulator":
                src = "built-in"
            else:
                src = f"provided by {ep.dist.name}"
        else:
            src = "unknown plugin source"

        doc = getattr(cls, "__doc__", "") or ""

        click.echo(f"{ep.name} ({src})", file=output)
        if doc:
            doc = doc.splitlines()[0]
            click.secho(f"    {doc}\n", italic=True, file=output)
        else:
            click.secho("    No description given.\n", italic=True, file=output)


@plugin.command
@click.argument("keyword", nargs=-1, required=True)
@click.option(
    "--type",
    "-t",
    "plugin_type",
    multiple=True,
    help=(
        "Restrict results to this type of plugin, e.g., openstb.simulator.distortion "
        "(or the shorter distortion for convenience). Can be given multiple times."
    ),
)
@click.option(
    "--name-only",
    "-n",
    is_flag=True,
    help="Only search the registered names of the plugins.",
)
@click.option(
    "--any",
    "match_any",
    is_flag=True,
    help=(
        "List plugins which match any of the keywords. The default behaviour is to "
        "only list those which match all keywords."
    ),
)
@click.option(
    "--case",
    "-c",
    is_flag=True,
    help="Make the search case-sensitive. By default it is case-insensitive.",
)
@click.option(
    "--output",
    "-o",
    type=click.File(mode="w"),
    default="-",
    show_default=False,
    help=(
        "Where to write the list of plugins. By default the list is written to the "
        "terminal."
    ),
)
def search(
    keyword: tuple[str],
    plugin_type: tuple[str],
    name_only: bool,
    match_any: bool,
    case: bool,
    output: TextIOWrapper,
):
    """Search for a plugin.

    This searches all plugins (or optionally, plugins of a certain type) for one or more
    keywords. By default, both the names and the help strings of the plugin are
    searched. Each result shows the name, source and type of the plugin along with the
    first line of its help string. The keywords are highlighted in the name and help
    string.

    Note that if a keyword only matches in the main body of the help string, the result
    may appear to be missing this keyword since only the first line is output.

    """

    # Find the entry points belonging to the specified groups.
    eps: list[tuple[importlib.metadata.EntryPoint, abc.Plugin]]
    if len(plugin_type) == 0:
        eps = [
            (ep, ep.load())
            for ep in importlib.metadata.entry_points()
            if ep.group.startswith("openstb.simulator.")
        ]
    else:
        type_set = {
            t if t.startswith("openstb.simulator.") else f"openstb.simulator.{t}"
            for t in plugin_type
        }
        eps = [
            (ep, ep.load())
            for ep in importlib.metadata.entry_points()
            if ep.group in type_set
        ]

    # Pre-process the keywords.
    if case:
        keywords = set(keyword)
    else:
        keywords = {k.lower() for k in keyword}

    def matches_keywords(ep: tuple[importlib.metadata.EntryPoint, abc.Plugin]) -> bool:
        """See if an entry point matches the keywords.

        Parameters
        ----------
        ep
            The entry point specification and implementing class.

        Returns
        -------
        bool
            True if it matches the keywords, False otherwise.

        """
        # Get the strings we are searching and normalise the case if required.
        name = ep[0].name
        doc = getattr(ep[1], "__doc__", "") or ""
        if not case:
            name = name.lower()
            doc = doc.lower()

        # Matching any keyword: return as soon as one matches.
        if match_any:
            for k in keywords:
                if k in name or (not name_only and k in doc):
                    return True
            return False

        # Matching all keywords: return as soon as one fails.
        else:
            for k in keywords:
                if k not in name and (name_only or k not in doc):
                    return False
            return True

    # Filter out plugins which don't match the keywords.
    eps = [ep for ep in eps if matches_keywords(ep)]

    def repl(match):
        return click.style(match.group(0), bold=True)

    # Define a regular expression to highlight the keywords in the result.
    if case:
        kw_re = re.compile(f"({'|'.join(keywords)})")
    else:
        kw_re = re.compile(f"({'|'.join(keywords)})", flags=re.IGNORECASE)

    # And now we can output the results.
    for ep, cls in eps:
        # Name, source and type of the plugin.
        name = kw_re.sub(repl, ep.name)
        if ep.dist is not None:
            if ep.dist.name == "openstb-simulator":
                src = "built-in"
            else:
                src = f"provided by {ep.dist.name}"
        else:
            src = "unknown plugin source"

        click.echo(f"{name} ({src})", file=output)
        click.secho(f"    Plugin type: {ep.group}", italic=True, file=output)

        # First line of the docstring if given.
        doc = getattr(cls, "__doc__", "") or ""
        if doc:
            doc = doc.splitlines()[0]
            click.echo(f"    {kw_re.sub(repl, doc)}", file=output)
        else:
            click.echo("    No description given.", file=output)
        click.echo(file=output)

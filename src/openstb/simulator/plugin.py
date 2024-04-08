# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import cast

from openstb.i18n.support import domain_translator

from openstb.simulator import abc


_ = domain_translator("openstb.simulator")


def installed_plugins(group: str) -> list[tuple[str, str]]:
    """List installed plugins.

    Parameters
    ----------
    group : str
        The name of the entry point group to list plugins for, e.g.,
        "openstb.simulator.trajectory".

    Returns
    -------
    installed : list
        A list of (name, src) tuples where name is the name the plugin is installed as
        and src is the reference to the module and class implementing the plugin.

    """
    return [(ep.name, ep.value) for ep in importlib.metadata.entry_points(group=group)]


def load_plugin(group: str, name: str) -> abc.Plugin:
    """Load a plugin class.

    Three formats are supported for the name of the plugin:

    1. The name of an installed plugin registered under the entry point group.

    2. A reference to an attribute name and the importable module it belongs to. The
       name "FromDisk:my_plugins.trajectories" effectively corresponds to this function
       being `from my_plugins.trajectories import FromDisk; return FromDisk`.

    3. The name of an attribute in a Python file and the path of the file. The name
       "FromDisk:/path/to/my_plugins.py" results in the function dynamically importing
       the `/path/to/my_plugins.py` file and returning the `FromDisk` attribute from it.
       Note that relative paths will be resolved relative to the current working
       directory.

    Parameters
    ----------
    group : str
        The name of the entry point group that the plugin belongs to, e.g.,
        "openstb.simulator.trajectory".
    name : str
        The user-specified name of the plugin.

    Returns
    -------
    plugin

    """
    classname, sep, modname_or_path = name.partition(":")

    # Separator not present: this is a registered plugin.
    if not sep:
        eps = importlib.metadata.entry_points(group=group, name=name)
        if not eps:
            raise ValueError(
                _("no {group} plugin named '{name}' is installed").format(
                    group=group, name=name
                )
            )
        if len(eps) > 1:
            raise ValueError(
                _("multiple {group} plugins named '{name}' are installed").format(
                    group=group, name=name
                )
            )
        return eps[0].load()

    # Not a registered plugin; does it refer to a file?
    path = Path(modname_or_path)
    if path.is_file():
        # Generate a unique module name. It is possible that two files at different
        # locations with the same stem could be requested; using just the stem could
        # result in the cached version of the first being used for the second.
        md5 = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
        modname = f"openstb.simulator.dynamic_plugins.{md5}.{path.stem}"

        # Generate an importlib spec for this module.
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            raise ValueError(
                _(
                    "Could not create import specification for plugin file {path}"
                ).format(path=path)
            )

        # Create and load the module.
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # If the attribute is missing, raise a better error as the default will include
        # the hashed module name we generated.
        try:
            return getattr(mod, classname)
        except AttributeError:
            raise AttributeError(
                _("file {path} has no attribute {classname}").format(
                    path=path, classname=classname
                )
            )

    # Assume it is an installed module.
    mod = importlib.import_module(modname_or_path)
    return getattr(mod, classname)


# Note: the cast() call does no runtime checking, simply returning the value unchanged.
# It is used to indicate to a static type checker that return values from the functions
# will have the specific type rather than the generic Plugin type.


def trajectory_plugin(name: str) -> abc.Trajectory:
    """Load a trajectory plugin.

    See the `load_plugin` function for details of the ``name`` parameter.

    """
    return cast(abc.Trajectory, load_plugin("openstb.simulator.trajectory", name))

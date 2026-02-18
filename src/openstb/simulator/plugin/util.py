# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from collections.abc import MutableMapping, Sequence
import inspect
from typing import Any, List, Literal, NotRequired, cast, get_args, get_origin
import warnings

from openstb.i18n.support import translations
from openstb.simulator.plugin import abc, loader
from openstb.simulator.types import SimulationConfig

trans = translations.load("openstb.simulator")
_ = trans.gettext
_n = trans.ngettext


def find_config_loader(
    source: str,
) -> tuple[str, type[abc.ConfigLoader]] | tuple[None, None]:
    """Try to find a config loader plugin.

    This is intended for use in a user interface. Given the source (e.g, the filename)
    of the desired configuration, it searches through the registered `ConfigLoader`
    plugins for one which says it might be able to handle it.

    Note that the plugins may return false positives or false negatives when asked about
    their ability to handle a source (see the `could_handle` method of the
    `ConfigLoader` class for details). If a class is returned, there is no guarantee it
    will actually be able to handle a source.

    Parameters
    ----------
    source : str
        The provided configuration source.

    Returns
    -------
    name : str, None
        The name of the plugin to use, or None if no plugin was found.
    cls : openstb.simulator.plugin.abc.ConfigLoader, None
        The plugin to use, or None if no plugin was found.

    """
    plugins = cast(
        list[tuple[str, str, type[abc.ConfigLoader]]],
        loader.registered_plugins("openstb.simulator.config_loader", load=True),
    )
    for name, _, cls in plugins:
        if cls.could_handle(source):
            return name, cls
    return None, None


def load_config_plugins(
    config_class: type[SimulationConfig] | None,
    config: MutableMapping[str, Any],
    missing: Literal["error", "warn", "ignore"] = "error",
    extra: Literal["error", "warn", "ignore"] = "error",
    unhandled_type: Literal["error", "warn", "ignore"] = "error",
    **keys,
):
    """Load configuration plugins.

    This uses a mapping of config keys to type annotations, either extracted from the
    configuration class or given as keyword arguments, to tell it what type of plugin
    is expected for each key. It then checks for a function registered with the
    `openstb.simulator.plugin.loader.register_loader` decorator to convert the entry in
    the configuration dictionary to a plugin instance.

    Parameters
    ----------
    config_class : type
        The class (not an instance of the class) specifying the configuration structure.
        This would typically be a subclass of `typing.TypedDict`, but can be any class
        which has type-annotated properties corresponding to the configuration entries.
    config : mapping
        The current configuration. This will be modified in-place.
    missing, extra, unhandled_type : {"error", "warn", "ignore"}
        How to handle missing, extra (unknown) entries in ``config``, and unhandled
        plugin types (no registered loader): raise an error, issue a warning or ignore
        them.
    **keys
        Keyword arguments can be given to map the keys in ``config`` to the expected
        plugin type, e.g., `load_config_plugins(..., signal=abc.Signal,
        trajectory=abc.Trajectory)`. If no keyword arguments are given, the type
        annotations of ``config_class`` are used as the keys.


    """
    # No keys: use the annotations of the configuration class.
    if not keys:
        if config_class is not None:
            keys = inspect.get_annotations(config_class)

    if not keys:
        raise ValueError(_("could not determine the configuration keys to process"))

    # Keys specified in the input. We will use this to track extra keys.
    config_keys = set(config.keys())

    for key, annotation in keys.items():
        required = True
        is_list = False

        # Get the unsubscripted type.
        origin = get_origin(annotation)

        # Entries of a TypedDict can be marked as not required. In this case, the first
        # subscripted argument is the type so pull that out, and then get its
        # unsubscripted version (we may have a NotRequired[list[...]] for example).
        if origin == NotRequired:
            required = False
            annotation = get_args(annotation)[0]
            origin = get_origin(annotation)

        # Expect a list. The first unsubscripted argument is the type within the list.
        if origin in (list, List):
            is_list = True
            annotation = get_args(annotation)[0]

        # Try to find a function to load the plugin.
        loader_func = loader.registered_loaders.get(annotation, None)
        if loader_func is None:
            if unhandled_type == "ignore":
                continue

            msg = _(
                "no loader available for plugin type {type} requested by section {key}"
            ).format(type=annotation.__name__, key=key)
            if unhandled_type == "warn":
                warnings.warn(msg, stacklevel=2)
                continue

            raise RuntimeError(msg)

        # No configuration entry for this key.
        if key not in config_keys:
            if not required or missing == "ignore":
                continue

            msg = _("section {key} missing from configuration").format(key=key)
            if missing == "warn":
                warnings.warn(msg, stacklevel=2)
                continue

            raise RuntimeError(msg)

        # A list is expected.
        if is_list:
            if not isinstance(config[key], Sequence):
                raise RuntimeError(
                    _("section {key} expects a list of plugins").format(key=key)
                )
            config[key] = [loader_func(c) for c in config[key]]

        # Single plugin.
        else:
            config[key] = loader_func(config[key])

        # Have handled this entry from the original configuration.
        config_keys.remove(key)

    # Configuration specified keys we don't know about.
    if config_keys and extra != "ignore":
        msg = _n(
            "extra section {keys} in configuration",
            "extra sections in configuration: {keys}",
            len(config_keys),
        ).format(keys=", ".join(config_keys))
        if extra == "error":
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

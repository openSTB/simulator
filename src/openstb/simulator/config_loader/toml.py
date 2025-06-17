# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
import tomllib

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import ConfigLoader
from openstb.simulator.types import PluginSpec

_ = translations.load("openstb.simulator").gettext


class TOMLLoader(ConfigLoader):
    """Load simulation configuration from a TOML file.

    The file may contain an entry ``include giving a list of other filenames to
    parse and merge into its configuration. This include behaviour is nested, i.e., any
    included file may also specify an `include` list. Any relative filenames are
    evaluated from the directory containing the file currently being processed.

    Included files may not overwrite values that have already been set. An exception is
    made for values from an array of tables; in this case, included tables are appended
    to the end of the current array.

    """

    def __init__(self, filename: os.PathLike[str] | str):
        """
        Parameters
        ----------
        filename
            The path to the configuration file.

        """
        self.filename = Path(filename)
        self._who_defined: dict[str, Path] = {}

    @classmethod
    def could_handle(cls, source: str) -> bool:
        # Assume any file ending with .toml could be loadable.
        return source.endswith(".toml")

    def load(self) -> dict:
        self._who_defined = {}
        return self._load_file(self.filename)

    def _load_file(self, filename: Path, current: dict | None = None) -> dict:
        """Load a file and update the current configuration.

        Parameters
        ----------
        filename : Path
            The path to the file to load.
        current : dict, optional
            The current configuration to update. If None, a new configuration will be
            started.

        Returns
        -------
        dict
            The current configuration after the file was loaded. If `current` was given
            to the method, it is both modified in-place and returned.

        """
        if current is None:
            self._who_defined = {}
            current = {}

        # Load this file.
        with filename.open("rb") as fp:
            config = tomllib.load(fp)

        # Extract any includes.
        includes = config.pop("include", [])
        if not isinstance(includes, list):
            raise ValueError(
                _("{filename}: value of include must be a list").format(
                    filename=str(filename)
                )
            )

        # Try to merge our config into the current.
        for entry, values in config.items():
            # We don't expect any other top-level entries. All values should be tables
            # (dicts) or arrays (lists) of tables. Collect the names and parameters into
            # a proper PluginSpec dictionary.
            if isinstance(values, dict):
                values = self._collect_parameters(filename, entry, values)
            elif isinstance(values, list):
                values = [
                    self._collect_parameters(filename, f"{entry}[{i}]", value)
                    for i, value in enumerate(values)
                ]
            else:
                raise ValueError(
                    _("{filename}: unexpected top-level entry {name}").format(
                        filename=filename, name=entry
                    )
                )

            # New key => easy.
            if entry not in current:
                current[entry] = values
                continue

            # Existing key is a list. We allow another list or a single table.
            if isinstance(current[entry], list):
                if isinstance(values, list):
                    current[entry].extend(values)
                elif isinstance(values, dict):
                    current[entry].append(values)
                else:
                    raise ValueError(
                        _("{filename}: {name} must be a table").format(
                            filename=filename, name=entry
                        )
                    )
                continue

            # Don't allow overwriting.
            msg = _("{filename} tried to overwrite value of {table} set by {original}")
            raise ValueError(
                msg.format(
                    filename=filename, original=self._who_defined[entry], table=entry
                )
            )

        # Update which tables came from which files. This will overwrite the source of
        # lists, but we don't need those.
        for entry in config.keys():
            self._who_defined[entry] = filename

        # Process any includes.
        for include in includes:
            includefn = Path(include)
            if not includefn.is_absolute():
                includefn = filename.parent / includefn
            self._load_file(includefn, current)

        return current

    def _collect_parameters(self, filename: Path, entry: str, spec: dict) -> PluginSpec:
        """Collect names and parameters into a PluginSpec dictionary.

        Parameters
        ----------
        filename : Path
            Filename the entry was loaded from. Used for error reporting and to set the
            source of the PluginSpec.
        entry : str
            Name of the table. Used for error reporting.
        spec : dict
            The values of the table loaded from the file.

        Returns
        -------
        dict
            A PluginSpec dictionary. The ``name`` entry from ``spec`` will be copied,
            and all other items will be placed into a dictionary under the
            ``parameters`` key.

        """
        name = spec.pop("name", None)
        if name is None:
            raise ValueError(
                _("{filename}: no plugin name given for entry {name}").format(
                    filename=filename, name=entry
                )
            )

        # Handle nested plugin specs (e.g., transducer beampatterns, signal windows).
        for k in spec.keys():
            if isinstance(spec[k], dict):
                spec[k] = self._collect_parameters(filename, f"{entry}.{k}", spec[k])

        return {
            "name": name,
            "parameters": spec,
            "spec_source": filename,
        }

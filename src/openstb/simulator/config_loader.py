# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
import tomllib

from openstb.i18n.support import domain_translator
from openstb.simulator.plugin.abc import ConfigLoader


_ = domain_translator("openstb.simulator", plural=False)


class TOMLLoader(ConfigLoader):
    """Load simulation configuration from a TOML file.

    The file may contain an entry ``__include__`` giving a list of other filenames to
    parse and merge into its configuration. This include behaviour is nested, i.e., any
    included file may also specify an `__include__` list. Any relative filenames are
    evaluated from the directory containing the file currently being processed.

    Included files may not overwrite values that have already been set. An exception is
    made for values from an array of tables; in this case, included tables are appended
    to the end of the current array.

    """

    def __init__(self, filename: os.PathLike[str] | str):
        """
        Parameters
        ----------
        filename : path-like
            The path to the configuration file.

        """
        self.filename = Path(filename)
        self._who_defined: dict[str, Path] = {}

    def load(self) -> dict:
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
        includes = config.pop("__include__", [])
        if not isinstance(includes, list):
            raise ValueError(
                _("{filename}: value of __include__ must be a list").format(
                    filename=str(filename)
                )
            )

        # Try to merge our config into the current.
        for k, v in config.items():
            # New key => easy.
            if k not in current:
                current[k] = v
                continue

            # Existing key is a list. We allow another list or a single table.
            if isinstance(current[k], list):
                if isinstance(v, list):
                    current[k].extend(v)
                elif isinstance(v, dict):
                    current[k].append(v)
                else:
                    raise ValueError(
                        _("{filename}: {name} must be a table").format(
                            filename=filename, name=k
                        )
                    )
                continue

            # Don't allow overwriting.
            msg = _("{filename} tried to overwrite value of {table} set by {original}")
            raise ValueError(
                msg.format(filename=filename, original=self._who_defined[k], table=k)
            )

        # Update which tables came from which files. This will overwrite the source of
        # lists, but we don't need those.
        for k in config.keys():
            self._who_defined[k] = filename

        # Process any includes.
        for include in includes:
            includefn = Path(include)
            if not includefn.is_absolute():
                includefn = filename.parent / includefn
            self._load_file(includefn, current)

        return current

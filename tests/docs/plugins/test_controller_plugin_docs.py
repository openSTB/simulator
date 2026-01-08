# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path

from openstb.simulator.config_loader.toml import TOMLLoader
from openstb.simulator.plugin import loader, util


def test_docs_plugins_controller_example(tmp_path):
    """docs: check example controller plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "controller" / "interface.md"
    example_md = example_file.read_text()

    # Extract the plugin contents and example configuration to files.
    with (
        (tmp_path / "example_controller.py").open("w") as fp,
        (tmp_path / "config.toml").open("w") as fc,
    ):
        in_block = False
        in_config = False
        config_done = False
        for line in example_md.splitlines():
            # Found the start of a Python block.
            if line.strip() == "```python":
                in_block = True
                continue

            # Only want the first TOML block.
            if line.strip() == "```toml" and not config_done:
                in_config = True
                continue

            # End of a block.
            if line.strip() == "```":
                in_block = False
                in_config = False
                continue

            if in_block:
                fp.write(line)
                fp.write("\n")

            if in_config:
                fc.write(line)
                fc.write("\n")
                config_done = True

    # Load the configuration.
    config = TOMLLoader(tmp_path / "config.toml").load()

    # And then run it in the temporary directory.
    original = os.getcwd()
    os.chdir(tmp_path)
    try:
        controller = loader.controller(config.pop("controller"))
        util.load_config_plugins(controller.config_class, config)
        controller.run(config)
    finally:
        os.chdir(original)

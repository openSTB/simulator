# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
import textwrap

import numpy as np

from openstb.simulator.config_loader.toml import TOMLLoader
from openstb.simulator.plugin import loader


def test_docs_plugins_specifying_examples(tmp_path):
    """docs: check examples in plugins/specifying.md"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "specifying.md"
    example_md = example_file.read_text()

    # Get the TOML and Python blocks.
    config = []
    code = []
    block = []
    in_config = False
    in_code = False
    for line in example_md.splitlines():
        if line.strip() == "```toml":
            in_config = True
            continue
        if line.strip() == "```python":
            in_code = True
            continue

        if line.strip() == "```":
            if in_config:
                config.append("\n".join(block))
            elif in_code:
                code.append("\n".join(block))

            in_config = False
            in_code = False
            block.clear()
            continue

        if in_config or in_code:
            block.append(line)

    # Write the from file test plugin to a temporary directory.
    (tmp_path / "my_trajectories.py").write_text(textwrap.dedent(code[0]))

    for i, cfgtext in enumerate(config):
        # Write the block configuration to a file and check it can be loaded.
        cfg_path = tmp_path / f"block_{i}.toml"
        cfg_path.write_text(cfgtext)
        cfg = TOMLLoader(cfg_path).load().pop("trajectory")

        # Move into the temporary directory (so the file can be found when needed) and
        # then create an instance of the plugin.
        original = os.getcwd()
        os.chdir(tmp_path)
        try:
            plugin = loader.trajectory(cfg)
        finally:
            os.chdir(original)

        # Check the class and parameters are as expected.
        if i == 2:
            assert plugin.__class__.__name__ == "Arc"
            assert np.allclose(plugin.start_angle, 0)
            assert np.allclose(plugin.end_angle, 90)
            assert np.allclose(plugin.radius, 50)
            assert np.allclose(plugin.speed, 1.2)
        else:
            assert plugin.__class__.__name__ == "Linear"
            assert np.allclose(plugin.start_position, [0, 0, 0])
            assert np.allclose(plugin.end_position, [100, 0, 0])
            assert np.allclose(plugin.velocity(0), [1.75, 0, 0])

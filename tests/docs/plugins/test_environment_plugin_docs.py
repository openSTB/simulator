# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.plugin import loader


def test_docs_plugin_environment_example(tmp_path: Path):
    """docs: check example environment plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "environment" / "interface.md"
    example_md = example_file.read_text()

    # Extract the example to a file.
    in_block = False
    fn = tmp_path / "example.py"
    with fn.open("w") as f:
        for line in example_md.splitlines():
            if line.strip() == "```python":
                in_block = True
                continue

            if line.strip() == "```":
                in_block = False
                continue

            if in_block:
                f.write(line)
                f.write("\n")

    # Load it.
    inst = loader.environment(
        {
            "name": f"LinearSpeed:{fn}",
            "parameters": {
                "surface_speed": 1480,
                "speed_gradient": -0.1,
                "temperature": 8.3,
                "salinity": 35.0,
            },
        }
    )

    t = [1, 2, 3]
    pos = [[0, 0, 0], [0, 0, 10], [0, 0, 20]]

    salinity = inst.salinity(t, pos)
    assert np.allclose(salinity, [35, 35, 35])
    temperature = inst.temperature(t, pos)
    assert np.allclose(temperature, [8.3, 8.3, 8.3])
    sound_speed = inst.sound_speed(t, pos)
    assert np.allclose(sound_speed, [1480, 1479, 1478])

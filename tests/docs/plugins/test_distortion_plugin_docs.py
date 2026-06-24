# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.plugin import loader


def test_docs_plugin_distortion_example(tmp_path: Path):
    """docs: check example distortion plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "distortion" / "interface.md"
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
    inst = loader.distortion(
        {
            "name": f"FrequencyDependentScaling:{fn}",
            "parameters": {
                "factor": 0.2,
            },
        }
    )

    # And apply it.
    freq = np.arange(50e3, 60e3, 10)
    S = np.ones((1, len(freq), 1), dtype=complex)
    Smod = inst.apply(0, freq, S, 55e3, None, (50e3, 60e3), None)  # type:ignore[arg-type]
    assert np.allclose(np.abs(Smod[0, :, 0]), 1 + ((freq - 50e3) * 0.2 / 9990))

# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.plugin import loader


def test_docs_plugin_signal_example(tmp_path: Path):
    """docs: check example signal plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "signal" / "interface.md"
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
    inst = loader.signal(
        {
            "name": f"TwoSinusoids:{fn}",
            "parameters": {
                "f1": 70e3,
                "f2": 60e3,
                "duration": 10e-3,
                "amplitude": 5,
            },
        }
    )

    # Check the properties.
    assert np.allclose(inst.duration, 10e-3)
    assert np.allclose(inst.maximum_frequency, 70e3)
    assert np.allclose(inst.minimum_frequency, 60e3)

    # And sample it.
    t = np.linspace(-2e-3, 12e-3, 500)
    s = inst.sample(t, 62e3)
    mask = (t >= 0) & (t <= 10e-3)
    assert np.allclose(np.abs(s[~mask]), 0)
    val = 5 * (np.exp(2j * np.pi * -2e3 * t[mask]) + np.exp(2j * np.pi * 8e3 * t[mask]))
    assert np.allclose(s[mask], val)

    # Also check with a 2D input.
    t2 = t.reshape(2, -1)
    s2 = inst.sample(t2, 62e3)
    mask2 = mask.reshape(2, -1)
    val2 = 5 * (np.exp(2j * np.pi * -2e3 * t2) + np.exp(2j * np.pi * 8e3 * t2))
    val2[~mask2] = 0
    assert np.allclose(s2, val2)

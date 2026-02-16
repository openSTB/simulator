# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.plugin import loader
from openstb.simulator.system.trajectory import Linear


def test_docs_plugin_ping_times_example(tmp_path: Path):
    """docs: check example ping time plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "ping_times" / "interface.md"
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
    inst = loader.ping_times(
        {
            "name": f"JitterPingTimes:{fn}",
            "parameters": {
                "seed": 4658716754,
                "mean_interval": 0.2,
                "jitter_std": 0.02,
            },
        }
    )

    # And check (with a suitable tolerance) the times are distributed as expected.
    traj = Linear([0, 0, 0], [300, 0, 0], 1)
    times = inst.calculate(traj)
    dt = np.diff(times)
    assert np.isclose(np.mean(dt), 0.2, atol=0, rtol=0.1)
    assert np.isclose(np.std(dt), 0.02, atol=0, rtol=0.5)

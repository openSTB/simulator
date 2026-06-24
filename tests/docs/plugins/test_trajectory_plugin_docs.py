# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.plugin import loader


def test_docs_plugin_trajectory_example(tmp_path: Path):
    """docs: check example trajectory plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "trajectory" / "interface.md"
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
    inst = loader.trajectory(
        {
            "name": f"TwoSegmentLinear:{fn}",
            "parameters": {
                "start_position": [0, 0, 0],
                "join_position": [10, 0, 0],
                "end_position": [10, 10, 0],
                "speed": 2,
            },
        }
    )

    # Check properties.
    assert np.allclose(inst.length, 20)
    assert np.allclose(inst.duration, 10)

    # Sample and check some values.
    t = [-1, 0, 4, 6, 12]

    pos = inst.position(t)
    assert pos.shape == (5, 3)
    assert np.all(np.isnan(pos[0]))
    assert np.allclose(pos[1], [0, 0, 0])
    assert np.allclose(pos[2], [8, 0, 0])
    assert np.allclose(pos[3], [10, 2, 0])
    assert np.all(np.isnan(pos[4]))

    ori = inst.orientation(t)
    assert ori.shape == (5, 4)
    assert np.all(np.isnan(ori[0]))  # type:ignore[call-overload]
    assert np.allclose(ori[1].ndarray, [1, 0, 0, 0])
    assert np.allclose(ori[2].ndarray, [1, 0, 0, 0])
    assert np.allclose(ori[3].ndarray, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.all(np.isnan(ori[4]))  # type:ignore[call-overload]

    vel = inst.velocity(t)
    assert vel.shape == (5, 3)
    assert np.all(np.isnan(vel[0]))
    assert np.allclose(vel[1], [2, 0, 0])
    assert np.allclose(vel[2], [2, 0, 0])
    assert np.allclose(vel[3], [0, 2, 0])
    assert np.all(np.isnan(vel[4]))

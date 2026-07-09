# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np

from openstb.simulator.environment.invariant import InvariantEnvironment
from openstb.simulator.plugin import loader


def test_docs_plugin_distortion_example(tmp_path: Path):
    """docs: check example distortion plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "scattering_model" / "interface.md"
    example_md = example_file.read_text()

    # Extract the example to a file.
    in_block = False
    fn = tmp_path / "example.py"
    with fn.open("w") as plugin_f:
        for line in example_md.splitlines():
            if line.strip() == "```python":
                in_block = True
                continue

            if line.strip() == "```":
                in_block = False
                continue

            if in_block:
                plugin_f.write(line)
                plugin_f.write("\n")

    # Load it.
    inst = loader.scattering_model(
        {
            "name": f"ConstantAmplitudeScattering:{fn}",
            "parameters": {
                "scale_factor": 0.2,
            },
        }
    )

    # Simple incident signal.
    t = np.arange(0, 0.01, 1 / 10e3)
    s = np.sin(2 * np.pi * 1e3 * t)
    s = s.reshape(-1, 1)

    # Evaluate at different angles.
    angle = np.radians(np.arange(1, 89, 0.5))
    x = np.zeros_like(angle)
    y = np.cos(angle)
    z = np.sin(angle)
    vec_i = np.stack((x, y, z), axis=-1)
    vec_s = np.stack((x, -y, -z), axis=-1)

    S = np.fft.fft(s, axis=0)
    f = np.fft.fftfreq(len(t), 1 / 10e3)

    env = InvariantEnvironment(sound_speed=1500, salinity=35, temperature=10)
    S_scat = inst.apply(
        f, S, 0, [0, 0, 0], [0, 0, -1], vec_i, vec_s, env, 1e3, (900, 1100)
    )

    # Output dimensions should match the input as the plugin doesn't need to expand the
    # target axis.
    assert S_scat.shape == S.shape

    # Intensity should change by the square of the given scale factor.
    s_scat = np.fft.ifft(S_scat, axis=0)
    I_inc = np.sum(np.abs(s) ** 2, axis=0)
    I_scat = np.sum(np.abs(s_scat) ** 2, axis=0)
    assert np.allclose(I_inc * 0.2**2, I_scat)

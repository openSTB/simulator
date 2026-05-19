# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import numpy as np
import zarr

from openstb.simulator.plugin import loader
from openstb.simulator.plugin.abc import ResultFormat


def test_docs_plugin_result_converter_example(tmp_path: Path):
    """docs: check example result converter plugin works"""
    base = Path(__file__).parent.parent.parent.parent
    example_file = base / "docs" / "plugins" / "result_converter" / "interface.md"
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
    inst = loader.result_converter(
        {
            "name": f"PressureNPYConverter:{fn}",
            "parameters": {
                "filename": str(tmp_path / "result.npy"),
            },
        }
    )

    # Generate some sample results.
    rng = np.random.default_rng(1646718461)
    Np, Nr, Ns = 3, 11, 200
    data = rng.normal(0, 1, (Np, Nr, Ns)) + 1j * rng.normal(0, 1, (Np, Nr, Ns))

    # Save it to a Zarr group.
    local_store = zarr.storage.MemoryStore()
    storage = zarr.create_group(store=local_store)
    pressure = storage.empty(
        name="pressure", shape=(Np, Nr, Ns), chunks=(Np, Nr, Ns), dtype="c16"
    )
    pressure[:] = data
    st = storage.empty(name="sample_time", shape=(Ns,), chunks=(Ns,), dtype="f8")
    st[:] = np.arange(Ns) / 40e3
    pst = storage.empty(name="ping_start_time", shape=(Np,), chunks=(Np,), dtype="f8")
    pst[:] = np.arange(Np)
    storage.attrs["baseband_frequency"] = 100e3
    storage.attrs["sample_rate"] = 40e3

    # Convert it.
    assert inst.can_handle(ResultFormat.ZARR_BASEBAND_PRESSURE, {})
    assert inst.convert(ResultFormat.ZARR_BASEBAND_PRESSURE, storage, {})

    # Load and check it.
    arr = np.load(tmp_path / "result.npy")
    assert np.allclose(arr, data)

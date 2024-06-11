# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np

from openstb.simulator import travel_time
from openstb.simulator.system.trajectory import Linear


def test_tt_stopandhop():
    """travel_time: stop-and-hop travel time calculations"""
    ttcalc = travel_time.StopAndHop(1500.0)

    # Trajectory along the y axis. At 1m/s, the system will be at (0, 0, 0) at time 10s.
    traj = Linear([-10, 0, 0], [10, 0, 0], 1.0)
    result = ttcalc.calculate(
        traj,
        10.0,
        [0, 1, 0],
        [[0, 1, 0], [0, 1, -1], [0, 1, 1]],
        [[0, 30, 0], [0, 40, 0]],
    )

    # Transmitter is at same x and z, so easy calculation.
    assert np.allclose(result.tx_position, [0, 1, 0])
    assert np.allclose(result.tx_vector, [0, 1, 0])
    assert np.allclose(result.tx_path_length, [29, 39])

    # First receiver is at same x and z, others at varying z.
    l0 = np.sqrt(29**2 + 1)
    l1 = np.sqrt(39**2 + 1)
    assert result.rx_position.shape == (3, 1, 3)  # not (3, 2, 3) but broadcastable
    assert np.allclose(result.rx_position, [[[0, 1, 0]], [[0, 1, -1]], [[0, 1, 1]]])
    assert np.allclose(result.rx_path_length, [[29, 39], [l0, l1], [l0, l1]])
    assert np.allclose(
        result.rx_vector,
        [
            [[0, -1, 0], [0, -1, 0]],
            [[0, -29 / l0, -1 / l0], [0, -39 / l1, -1 / l1]],
            [[0, -29 / l0, 1 / l0], [0, -39 / l1, 1 / l1]],
        ],
    )

    assert np.allclose(
        result.travel_time,
        [
            [58 / 1500.0, 78 / 1500.0],
            [(29 + l0) / 1500.0, (39 + l1) / 1500.0],
            [(29 + l0) / 1500.0, (39 + l1) / 1500.0],
        ],
    )
    assert result.scale_factor is None

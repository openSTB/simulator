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
    tt, tx_vec, tx_pathlen, rx_vec, rx_pathlen, scale = ttcalc.calculate(
        traj,
        10.0,
        [0, 1, 0],
        [[0, 1, 0], [0, 1, -1], [0, 1, 1]],
        [[0, 30, 0], [0, 40, 0]],
    )

    # Transmitter is at same x and z, so easy calculation.
    assert np.allclose(tx_vec, [0, 1, 0])
    assert np.allclose(tx_pathlen, [29, 39])

    # First receiver is at same x and z, others at varying z.
    l0 = np.sqrt(29**2 + 1)
    l1 = np.sqrt(39**2 + 1)
    assert np.allclose(rx_pathlen, [[29, 39], [l0, l1], [l0, l1]])
    assert np.allclose(
        rx_vec,
        [
            [[0, -1, 0], [0, -1, 0]],
            [[0, -29 / l0, -1 / l0], [0, -39 / l1, -1 / l1]],
            [[0, -29 / l0, 1 / l0], [0, -39 / l1, 1 / l1]],
        ],
    )

    assert np.allclose(
        tt,
        [
            [58 / 1500.0, 78 / 1500.0],
            [(29 + l0) / 1500.0, (39 + l1) / 1500.0],
            [(29 + l0) / 1500.0, (39 + l1) / 1500.0],
        ],
    )
    assert scale is None

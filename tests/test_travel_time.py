# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from datetime import datetime

import numpy as np
import pytest
import quaternionic

from openstb.simulator import travel_time
from openstb.simulator.environment import InvariantEnvironment
from openstb.simulator.plugin.abc import Trajectory
from openstb.simulator.system.trajectory import Linear


def test_tt_stopandhop():
    """travel_time: stop-and-hop travel time calculations"""
    ttcalc = travel_time.StopAndHop()

    # Trajectory along the y axis. At 1m/s, the system will be at (0, 0, 0) at time 10s.
    traj = Linear([-10, 0, 0], [10, 0, 0], 1.0)
    result = ttcalc.calculate(
        traj,
        10.0,
        InvariantEnvironment(salinity=35, sound_speed=1500.0, temperature=9.6),
        [0, 1, 0],
        [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
        [[0, 1, 0], [0, 1, -1], [0, 1, 1]],
        [[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]] * 3,
        [[0, 30, 0], [0, 40, 0]],
    )

    # Transmitter is at same x and z, so easy calculation.
    assert np.allclose(result.tx_position, [0, 1, 0])
    assert np.allclose(
        np.array(result.tx_orientation), [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]
    )
    assert np.allclose(result.tx_velocity, [1, 0, 0])
    assert np.allclose(result.tx_vector, [0, 1, 0])
    assert np.allclose(result.tx_path_length, [29, 39])

    # First receiver is at same x and z, others at varying z.
    l0 = np.sqrt(29**2 + 1)
    l1 = np.sqrt(39**2 + 1)
    assert result.rx_position.shape == (3, 1, 3)  # not (3, 2, 3) but broadcastable
    assert np.allclose(result.rx_position, [[[0, 1, 0]], [[0, 1, -1]], [[0, 1, 1]]])
    assert result.rx_orientation.shape == (3, 1, 4)
    assert np.allclose(
        np.array(result.rx_orientation), [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]
    )
    assert result.rx_velocity.shape == (1, 1, 3)  # not (3, 2, 3) but broadcastable.
    assert np.allclose(result.rx_velocity, [1, 0, 0])
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


def test_tt_iterative_linear_colocated():
    """travel_time: iterative calculation, colocated system + linear trajectory"""
    c = 1478.0
    v = 1.0

    ttcalc = travel_time.Iterative(max_iterations=20, tolerance=50e-9)
    traj = Linear([-10, 0, 0], [10, 0, 0], v)
    result = ttcalc.calculate(
        traj,
        10.0,
        InvariantEnvironment(salinity=18, sound_speed=c, temperature=4.2),
        [0, 1, 0],
        [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
        [[0, 1, 0]],
        [[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]],
        [[0, 30, 0], [0, 40, 0]],
    )

    # Check transmitter details.
    assert result.tx_position.shape == (3,)
    assert np.allclose(result.tx_position, [0, 1, 0])
    assert result.tx_orientation.shape == (4,)
    assert np.allclose(
        np.array(result.tx_orientation), [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]
    )
    assert result.tx_velocity.shape == (3,)
    assert np.allclose(result.tx_velocity, [1, 0, 0])
    assert result.tx_vector.shape == (2, 3)
    assert np.allclose(result.tx_vector, [0, 1, 0])
    assert result.tx_path_length.shape == (2,)
    assert np.allclose(result.tx_path_length, [29, 39])

    # True travel time for the linear velocity.
    dotprod = np.sum(result.tx_velocity[np.newaxis, :] * result.tx_vector, axis=-1)
    twtt = (2 * result.tx_path_length * c - dotprod) / (c**2 - v**2)
    assert result.travel_time.shape == (1, 2)
    assert np.allclose(result.travel_time, twtt)

    # Orientation and velocity are constant.
    assert result.rx_orientation.shape == (1, 2, 4)
    assert np.allclose(
        np.array(result.rx_orientation), [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]
    )
    assert result.rx_velocity.shape == (1, 2, 3)
    assert np.allclose(result.rx_velocity, [1, 0, 0])

    # Receiver positions should differ.
    assert result.rx_position.shape == (1, 2, 3)
    assert np.allclose(result.rx_position, [[twtt[0] * v, 1, 0], [twtt[1] * v, 1, 0]])
    assert result.rx_vector.shape == (1, 2, 3)

    # Path length and direction.
    rvec = np.array([[twtt[0] * v, -29, 0], [twtt[1] * v, -39, 0]])
    rlen = np.linalg.norm(rvec, axis=-1)
    assert result.rx_path_length.shape == (1, 2)
    assert np.allclose(result.rx_path_length, rlen)
    assert result.rx_vector.shape == (1, 2, 3)
    assert np.allclose(result.rx_vector, rvec / rlen[:, np.newaxis])


class _RandomPositionTraj(Trajectory):
    duration = 10.0
    length = 10.0
    start_time = datetime.now()

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def position(self, t):
        shape = list(np.array(t).shape) + [3]
        return self.rng.uniform(-10, 10, shape)

    def orientation(self, t):
        shape = list(np.array(t).shape) + [4]
        ori = np.zeros(shape, dtype=float)
        ori[..., 0] = 1
        return quaternionic.array(ori)

    def velocity(self, t):
        shape = list(np.array(t).shape) + [3]
        vel = np.zeros(shape, dtype=float)
        vel[..., 0] = 1
        return vel


def test_tt_iterative_nonconverging():
    """travel_time: iterative calculation reports non-convergence"""
    ttcalc = travel_time.Iterative(max_iterations=20, tolerance=50e-9)
    traj = _RandomPositionTraj(1178191)
    with pytest.raises(RuntimeError, match="could not converge"):
        ttcalc.calculate(
            traj,
            10.0,
            InvariantEnvironment(salinity=18, sound_speed=1492.4, temperature=4.2),
            [0, 1, 0],
            [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
            [[0, 1, 0]],
            [[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]],
            [[0, 30, 0], [0, 40, 0]],
        )

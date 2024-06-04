# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import quaternionic

from openstb.simulator.system import trajectory


@pytest.mark.parametrize("cls", [trajectory.Linear])
def test_system_trajectory_start_time(cls):
    """system.trajectory: trajectory plugin start time handling"""
    if cls == trajectory.Linear:
        params = {
            "start_position": [0, 0, 0],
            "end_position": [100, 0, 0],
            "speed": 2.0,
        }

    # Default should be None -> current time.
    traj = cls(**params)
    assert (datetime.now(timezone.utc) - traj.start_time).seconds == pytest.approx(0)
    traj = cls(**params, start_time=None)
    assert (datetime.now(timezone.utc) - traj.start_time).seconds == pytest.approx(0)

    # Explicit timestamp in UTC.
    start = datetime(2023, 12, 11, 10, 59, 33, tzinfo=timezone.utc)
    traj = cls(**params, start_time=start)
    assert traj.start_time == start

    # Explicit timestamp in different timezone should be converted to UTC.
    cst = timezone(timedelta(hours=1))
    start_cst = datetime(2023, 12, 11, 11, 59, 33, tzinfo=cst)
    traj = cls(**params, start_time=start_cst)
    assert traj.start_time == start

    # Integer corresponds to seconds from midnight 1/1/1970 in UTC.
    traj = cls(**params, start_time=int(start.timestamp()))
    assert traj.start_time == start

    # String in ISO 8601 format, with or without timezone information.
    traj = cls(**params, start_time="2023-12-11T10:59:33")
    assert traj.start_time == start
    traj = cls(**params, start_time="2023-12-11T10:59:33+00:00")
    assert traj.start_time == start
    traj = cls(**params, start_time="2023-12-11T22:59:33+12:00")
    assert traj.start_time == start


def test_system_trajectory_linear_errors():
    """system.trajectory.Linear: error handling"""
    with pytest.raises(ValueError, match="3 element vector required.+start"):
        trajectory.Linear(0, [100, 0, 0], 1.5)
    with pytest.raises(ValueError, match="3 element vector required.+start"):
        trajectory.Linear([0, 0], [100, 0, 0], 1.5)
    with pytest.raises(ValueError, match="3 element vector required.+start"):
        trajectory.Linear([[0, 0, 0], [0, 0, 0]], [100, 0, 0], 1.5)

    with pytest.raises(ValueError, match="3 element vector required.+end"):
        trajectory.Linear([0, 0, 0], 100, 1.5)
    with pytest.raises(ValueError, match="3 element vector required.+end"):
        trajectory.Linear([0, 0, 0], [100, 0], 1.5)
    with pytest.raises(ValueError, match="3 element vector required.+end"):
        trajectory.Linear([0, 0, 0], [[100, 0, 0], [100, 0, 0]], 1.5)

    with pytest.raises(ValueError, match="speed of linear.+positive"):
        trajectory.Linear([0, 0, 0], [100, 0, 0], -1.5)
    with pytest.raises(ValueError, match="speed of linear.+positive"):
        trajectory.Linear([0, 0, 0], [100, 0, 0], 0)


def test_system_trajectory_linear_properties():
    """system.trajectory.Linear: trajectory properties"""
    traj = trajectory.Linear([0, 0, 0], [100, 0, 0], 2.0)
    assert traj.length == pytest.approx(100)
    assert traj.duration == pytest.approx(50)

    traj = trajectory.Linear([-50, -3, 7.5], [60, -8, 8], 1.8)
    assert traj.length == pytest.approx(110.11471)
    assert traj.duration == pytest.approx(61.17484)


def test_system_trajectory_linear_position():
    """system.trajectory.Linear: trajectory position"""
    traj = trajectory.Linear([0, 0, 0], [100, 0, 0], 2.0)
    t = np.arange(-3, 53)
    pos = traj.position(t)
    assert np.all(np.isnan(pos[:3, :]))
    assert np.all(np.isnan(pos[-2:, :]))
    assert np.allclose(pos[3:-2, 1:], 0)
    assert np.allclose(pos[3:-2, 0], np.arange(0, 101, 2))

    traj = trajectory.Linear([100, 0, 0], [0, 0, 0], 2.0)
    t = np.arange(-3, 53)
    pos = traj.position(t)
    assert np.all(np.isnan(pos[:3, :]))
    assert np.all(np.isnan(pos[-2:, :]))
    assert np.allclose(pos[3:-2, 1:], 0)
    assert np.allclose(pos[3:-2, 0], np.arange(100, -1, -2))

    traj = trajectory.Linear([30, -10, 6], [30, 50, 6], 1)
    t = np.arange(10, 40).reshape(10, 3)
    pos = traj.position(t)
    assert pos.shape == (10, 3, 3)
    assert np.allclose(pos[..., 0], 30)
    assert np.allclose(pos[..., 1], np.arange(0, 30).reshape(10, 3))
    assert np.allclose(pos[..., 2], 6)


@pytest.mark.parametrize(
    "end,quat",
    [
        [[1, 0, 0], [1, 0, 0, 0]],
        [[-1, 0, 0], [0, 0, 0, 1]],
        [[0, 1, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]],
        [[0, -1, 0], [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]],
        [[0, 0, 1], [np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]],
        [[0, 0, -1], [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]],
        [[1, 1, 0], [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]],
        [[-1, 1, 0], [np.cos(3 * np.pi / 8), 0, 0, np.sin(3 * np.pi / 8)]],
        [[-1, -1, 0], [np.cos(-3 * np.pi / 8), 0, 0, np.sin(-3 * np.pi / 8)]],
        [[1, 0, 1], [np.cos(-np.pi / 8), 0, np.sin(-np.pi / 8), 0]],
        [[1, 0, -1], [np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0]],
    ],
)
def test_system_trajectory_linear_orientation_single(end, quat):
    """system.trajectory.Linear: trajectory orientation (single values)"""
    traj = trajectory.Linear([0, 0, 0], end, 2.0)
    ori = traj.orientation(0)
    quat = quaternionic.array(quat)
    assert np.allclose(ori, quat) or np.allclose(ori, -quat)


def test_system_trajectory_linear_orientation_array():
    """system.trajectory.Linear: trajectory orientation (full array)"""
    traj = trajectory.Linear([0, 0, 0], [100, 0, 0], 2.0)
    t = np.arange(-3, 53)
    ori = traj.orientation(t)
    assert np.all(np.isnan(ori[:3, :]))
    assert np.all(np.isnan(ori[-2:, :]))
    expected = quaternionic.one
    assert np.allclose(ori[3:-2], expected) or np.allclose(ori[3:-2], -expected)

    traj = trajectory.Linear([100, 0, 0], [0, 0, 0], 2.0)
    t = np.arange(-3, 53)
    ori = traj.orientation(t)
    assert np.all(np.isnan(ori[:3, :]))
    assert np.all(np.isnan(ori[-2:, :]))
    expected = quaternionic.array([0, 0, 0, 1])
    assert np.allclose(ori[3:-2], expected) or np.allclose(ori[3:-2], -expected)

    traj = trajectory.Linear([30, -10, 6], [30, 50, 6], 1)
    t = np.arange(1, 21).reshape(4, 5)
    ori = traj.orientation(t)
    assert ori.shape == (4, 5, 4)
    expected = quaternionic.array.from_rotation_vector([0, 0, np.pi / 2])
    assert np.allclose(ori[3:-2], expected) or np.allclose(ori[3:-2], -expected)


def test_system_trajectory_linear_velocity():
    """system.trajectory.Linear: trajectory velocity"""
    traj = trajectory.Linear([0, 0, 0], [100, 0, 0], 2.0)
    t = np.arange(-3, 53)
    vel = traj.velocity(t)
    assert np.all(np.isnan(vel[:3, :]))
    assert np.all(np.isnan(vel[-2:, :]))
    assert np.allclose(vel[3:-2], [2.0, 0, 0])

    traj = trajectory.Linear([100, 0, 0], [0, 0, 0], 2.0)
    t = np.arange(-3, 53)
    vel = traj.velocity(t)
    assert np.all(np.isnan(vel[:3, :]))
    assert np.all(np.isnan(vel[-2:, :]))
    assert np.allclose(vel[3:-2], [-2.0, 0, 0])

    traj = trajectory.Linear([30, -10, 6], [30, 50, 6], 1)
    t = np.arange(1, 21).reshape(4, 5)
    vel = traj.velocity(t)
    assert vel.shape == (4, 5, 3)
    assert np.allclose(vel, [0, 1, 0])

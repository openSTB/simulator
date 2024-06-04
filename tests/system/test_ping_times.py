# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest

from openstb.simulator.system import ping_times, trajectory


def test_system_ping_times_interval_errors():
    """system.ping_times.ConstantInterval: error handling"""
    with pytest.raises(ValueError, match="ping interval must be greater than zero"):
        ping_times.ConstantInterval(-2, 1.5, 2.0)
    with pytest.raises(ValueError, match="ping interval must be greater than zero"):
        ping_times.ConstantInterval(0, 1.5, 2.0)

    with pytest.raises(ValueError, match="delay of first .+ less than zero"):
        ping_times.ConstantInterval(0.5, -0.1, 2.0)
    with pytest.raises(ValueError, match="delay of last .+ less than zero"):
        ping_times.ConstantInterval(0.5, 1.5, -0.01)


def test_system_ping_times_interval_calculate():
    """system.ping_times.ConstantInterval: ping time calculation"""
    # No delays.
    const_interval = ping_times.ConstantInterval(0.2, 0, 0)
    traj = trajectory.Linear([0, 0, 0], [75, 0, 0], 1.5)
    times = const_interval.calculate(traj)
    assert np.allclose(times, np.arange(0, 50, 0.2))

    # Start delay only.
    const_interval = ping_times.ConstantInterval(0.2, 2.5, 0)
    traj = trajectory.Linear([10, 0, 5], [10, 75, 5], 1.5)
    times = const_interval.calculate(traj)
    assert np.allclose(times, np.arange(2.5, 50, 0.2))

    # Start and end delay.
    const_interval = ping_times.ConstantInterval(0.35, 2.5, 5.0)
    traj = trajectory.Linear([10, -8, 5], [10, 106, 5], 1.5)
    times = const_interval.calculate(traj)
    assert np.allclose(times, np.arange(2.5, 71, 0.35))


def test_system_ping_times_distance_errors():
    """system.ping_times.ConstantDistance: error handling"""
    with pytest.raises(ValueError, match="ping-to-ping distance .+ greater than zero"):
        ping_times.ConstantDistance(-1, 0, 0)
    with pytest.raises(ValueError, match="ping-to-ping distance .+ greater than zero"):
        ping_times.ConstantDistance(0, 0, 0)

    with pytest.raises(ValueError, match="start offset cannot be less than zero"):
        ping_times.ConstantDistance(0.3, -0.1, 3)
    with pytest.raises(ValueError, match="end offset cannot be less than zero"):
        ping_times.ConstantDistance(0.3, 5, -0.02)

    with pytest.raises(ValueError, match="sampling factor cannot be less than one"):
        ping_times.ConstantDistance(0.3, 3, 5, 0.9)
    with pytest.raises(ValueError, match="sampling factor cannot be less than one"):
        ping_times.ConstantDistance(0.3, 3, 5, 0)
    with pytest.raises(ValueError, match="sampling factor cannot be less than one"):
        ping_times.ConstantDistance(0.3, 3, 5, -2)


def test_system_ping_times_distance_calculate():
    """system.ping_times.ConstantDistance: ping time calculation"""
    # No offsets.
    const_dist = ping_times.ConstantDistance(0.4, 0, 0)
    traj = trajectory.Linear([10, 0, 0], [-90, 0, 0], 1.6)
    times = const_dist.calculate(traj)
    assert np.allclose(times, np.arange(0, 100 / 1.6, 0.25))

    # Start offset only.
    const_dist = ping_times.ConstantDistance(0.3, 5, 0)
    traj = trajectory.Linear([0, 0, 5], [75, 75, 5], 1.4)
    times = const_dist.calculate(traj)
    assert np.allclose(times, np.arange(5 / 1.4, np.sqrt(2) * 75 / 1.4, 0.3 / 1.4))

    # Start and end offsets.
    const_dist = ping_times.ConstantDistance(0.3, 5, 6)
    traj = trajectory.Linear([30, -30, 6], [30, 70, 6], 2.0)
    times = const_dist.calculate(traj)
    assert np.allclose(times, np.arange(2.5, 47, 0.15))

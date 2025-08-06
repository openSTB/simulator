# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np
import pytest
from scipy.stats import kstest

from openstb.simulator.target import points


def test_target_points_rprect_error():
    """target.points: RandomPointRect error handling"""
    with pytest.raises(ValueError, match="centre position .+ exactly 3 values"):
        points.RandomPointRect(1, 10, 10, [0, 0], [0, 0, -1], 5, 1)
    with pytest.raises(ValueError, match="centre position .+ exactly 3 values"):
        points.RandomPointRect(1, 10, 10, [0, 0, 0, 0], [0, 0, -1], 5, 1)

    with pytest.raises(ValueError, match="normal .+ exactly 3 values"):
        points.RandomPointRect(1, 10, 10, [0, 0, 0], [0, 0], 5, 1)
    with pytest.raises(ValueError, match="normal .+ exactly 3 values"):
        points.RandomPointRect(1, 10, 10, [0, 0, 0], [0, 0, -1, 0], 5, 1)
    with pytest.raises(ValueError, match="length of .+ normal cannot be zero"):
        points.RandomPointRect(1, 10, 10, [0, 0, 0], [0, 0, 0], 5, 1)

    with pytest.raises(ValueError, match="unexpected value .+ for reflectivity"):
        points.RandomPointRect(1, 10, 10, [0, 0, 0], [0, 0, -1], 5, "unknown")


@pytest.mark.parametrize(
    "Dx,Dy,density,expected",
    [
        (10, 10, 1, 100),
        (100, 100, 1.5, 15000),
        (55.5, 10.1, 17, 9529),
        (18.6, 29.4, 13, 7109),
    ],
)
def test_target_points_rprect_len(Dx, Dy, density, expected):
    """target.points: RandomPointRect number of points"""
    tgt = points.RandomPointRect(11091, Dx, Dy, [0, 0, 0], [0, 0, -1], density, 1)
    tgt.prepare()
    assert len(tgt) == expected


def test_target_points_rprect_uniform():
    """target.points: RandomPointRect uniformly distributes points"""
    tgt = points.RandomPointRect(67191, 100, 50, (0, 0, 0), (0, 0, -1), 10, 1)
    tgt.prepare()

    # Use a Kolmogorov-Smirnov test with a confidence level of 90%. Note that for the
    # kstest() function the uniform distribution is parametrised as (min, width).
    stat, pvalue = kstest(tgt.position[:, 0], "uniform", (-50, 100))
    assert pvalue > 0.1, "x positions not uniformly distributed"
    stat, pvalue = kstest(tgt.position[:, 1], "uniform", (-25, 50))
    assert pvalue > 0.1, "y positions not uniformly distributed"


@pytest.mark.parametrize(
    "seed,Dx,Dy,centre,normal",
    [
        (177161, 10, 10, [0, 0, 0], [0, 0, -1]),
        (22422242, 10, 10, [0, 0, 0], [0, 0, 1]),
        (17181920, 100, 30, [50, 45, 10.2], [0, 0.1, -1]),
        (17181920, 100, 30, [50, 45, 10.2], [0.2, 0, -1]),
        (54326711, 45, 100, [0, 50, 7.8], [-0.2, 0.1, -1]),
    ],
)
def test_target_points_rprect_position(seed, Dx, Dy, centre, normal):
    """target.points: RandomPointRect positions"""
    tgt = points.RandomPointRect(seed, Dx, Dy, centre, normal, 10, 1)
    tgt.prepare()

    # Check the centroid of the points. We need to have a reasonable tolerance here.
    assert np.allclose(tgt.position.mean(axis=0), centre, rtol=0, atol=0.2)

    # Take the general form of the plane equation, ax + by + cz + d = 0 where (a, b, c)
    # is the normal and d a constant. Rearrange to -z = (a/c)x + (b/c)y + (d/c).
    # From this we can form a system of equations Ap = z where A = [x y 1] and
    # p = [a/c b/c d/c].
    A = np.ones((len(tgt), 3), dtype=float)
    A[:, :2] = tgt.position[:, :2]
    z = tgt.position[:, 2]

    # Solve for p using the pseudoinverse form of least squares.
    p = np.linalg.inv(A.T @ A) @ A.T @ z

    # Error to each measurement.
    errors = z - A @ p
    assert np.allclose(errors, 0), "points do not lie in a plane"

    # We do not care about d; this is an offset influenced by the centre. We need to
    # check that the first two components of the solution correspond to (a/c, b/c).
    expected = np.array([normal[0] / normal[2], normal[1] / normal[2]])
    assert np.allclose(p[:2], expected), "points lie in wrong plane"


@pytest.mark.parametrize(
    "param,expected",
    [
        (0.5, 0.5),
        ("omnidirectional", 1 / (4 * np.pi)),
        ("hemispherical", 1 / (2 * np.pi)),
    ],
)
def test_target_points_rprect_reflectivity(param, expected):
    """target.points: RandomPointRect reflectivity settings"""
    tgt = points.RandomPointRect(1, 10, 10, [0, 0, 0], [0, 0, -1], 5, param)
    tgt.prepare()
    assert np.allclose(tgt.reflectivity, expected)


def test_target_points_rprect_repeatability():
    """target.points: RandomPointRect targets are repeatable"""
    tgt1 = points.RandomPointRect(1776771, 10, 44, [0, 0, 0], [0, 0, -1], 5, 0.3)
    tgt1.prepare()

    tgt2 = points.RandomPointRect(1776771, 10, 44, [0, 0, 0], [0, 0, -1], 5, 0.3)
    tgt2.prepare()

    # Could probably used == with the position/reflectivity, but I don't trust floats.
    assert len(tgt1) == len(tgt2)
    assert np.allclose(tgt1.position, tgt2.position)
    assert np.allclose(tgt1.reflectivity, tgt2.reflectivity)


@pytest.mark.parametrize(
    "seed,p1,p2,p3",
    [
        (1178167, [0, 0, 0], [1, 0, 0], [0, 1, 0]),
        (8984019, [1, 0, 1], [1, 1, 2], [2, 1, 3]),
        (6629369, [0.5, 1.25, -10.75], [2.22, 3.33, -8.9], [-0.5, 1.3, -7.7]),
    ],
)
def test_target_points_rptri_basic(seed, p1, p2, p3):
    """target.points: basic behaviour of RandomPointTriangle"""
    tri = points.RandomPointTriangle(seed, p1, p2, p3, 10, 0.1)
    tri.prepare()

    # Normal for the plane containing the triangle.
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    assert not np.isclose(np.linalg.norm(normal), 0)

    # Check the number of points.
    area = 0.5 * np.sqrt(np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) ** 2)
    N = int(np.round(area * 10))
    assert len(tri) == N

    # Check all points lie in the plane.
    assert tri.position.shape == (N, 3)
    oop = np.dot(tri.position - p1, normal)
    assert np.allclose(oop, 0), "not all points in same plane as triangle"

    # Shift the corners so each target becomes the origin of the whole triangle.
    a = p1 - tri.position
    b = p2 - tri.position
    c = p3 - tri.position

    # Find the normals of the three sub-triangles formed with each target.
    u = np.cross(b, c)
    v = np.cross(c, a)
    w = np.cross(a, b)

    # For a point in the triangle, all normals will point in the same direction.
    assert np.all(np.sum(u * v, axis=-1) >= 0), "points not in triangle"
    assert np.all(np.sum(u * w, axis=-1) >= 0), "points not in triangle"

    # Check the reflectivity is the value we set.
    assert tri.reflectivity.shape == (N,)
    assert np.allclose(tri.reflectivity, 0.1)


def test_target_points_rptri_corner_error():
    """target.points: error checking of corners in RandomPointTriangle"""
    with pytest.raises(ValueError, match="p1.+three values"):
        points.RandomPointTriangle(110110, [0, 1], [1, 2, 3], [4, 5, 6], 5, 0.2)
    with pytest.raises(ValueError, match="p1.+three values"):
        points.RandomPointTriangle(110110, [0, 1, 2, 3], [1, 2, 3], [4, 5, 6], 5, 0.2)
    with pytest.raises(ValueError, match="p1.+three values"):
        points.RandomPointTriangle(
            110110, [[0, 1, 0], [0, 0, 1]], [1, 2, 3], [4, 5, 6], 5, 0.2
        )

    with pytest.raises(ValueError, match="p2.+three values"):
        points.RandomPointTriangle(110110, [1, 2, 3], [0, 1], [4, 5, 6], 5, 0.2)
    with pytest.raises(ValueError, match="p2.+three values"):
        points.RandomPointTriangle(110110, [1, 2, 3], [0, 1, 2, 3], [4, 5, 6], 5, 0.2)
    with pytest.raises(ValueError, match="p2.+three values"):
        points.RandomPointTriangle(
            110110, [1, 2, 3], [[0, 1, 0], [0, 0, 1]], [4, 5, 6], 5, 0.2
        )

    with pytest.raises(ValueError, match="p3.+three values"):
        points.RandomPointTriangle(110110, [1, 2, 3], [4, 5, 6], [0, 1], 5, 0.2)
    with pytest.raises(ValueError, match="p3.+three values"):
        points.RandomPointTriangle(110110, [1, 2, 3], [4, 5, 6], [0, 1, 2, 3], 5, 0.2)
    with pytest.raises(ValueError, match="p3.+three values"):
        points.RandomPointTriangle(
            11, [1, 2, 3], [4, 5, 6], [[0, 1, 0], [0, 0, 1]], 5, 0.2
        )

    with pytest.raises(ValueError, match="p1.+p2.+coincident"):
        points.RandomPointTriangle(10, [0, 1, 0], [0, 1, 0], [1, 2, 3], 5, 0.2)
    with pytest.raises(ValueError, match="p1.+p3.+coincident"):
        points.RandomPointTriangle(10, [-10, 11, 2], [1, 2, 3], [-10, 11, 2], 5, 0.2)
    with pytest.raises(ValueError, match="cannot be colinear"):
        points.RandomPointTriangle(10, [0, 0, 0], [1, 0, 0], [10, 0, 0], 8, 0.1)
    with pytest.raises(ValueError, match="cannot be colinear"):
        points.RandomPointTriangle(10, [0, 0, 0], [1, 0, 0], [-10, 0, 0], 8, 0.1)


def test_target_points_rptri_error():
    """target.points: other error checking with RandomPointTriangle"""
    with pytest.raises(ValueError, match="density cannot be negative"):
        points.RandomPointTriangle(110110, [0, 0, 1], [1, 2, 3], [4, 5, 6], -5, 0.2)
    with pytest.raises(ValueError, match="no point targets"):
        points.RandomPointTriangle(110110, [0, 0, 1], [1, 2, 3], [4, 5, 6], 0.0001, 0.2)


@pytest.mark.parametrize(
    "param,expected",
    [
        (0.5, 0.5),
        ("omnidirectional", 1 / (4 * np.pi)),
        ("hemispherical", 1 / (2 * np.pi)),
    ],
)
def test_target_points_rptri_reflectivity(param, expected):
    """target.points: RandomPointTriangle reflectivity settings"""
    tgt = points.RandomPointTriangle(32, [0, 0, 0], [1, 0, 0], [0, 0, 1], 5, param)
    tgt.prepare()
    assert np.allclose(tgt.reflectivity, expected)


def test_target_points_rptri_repeatability():
    """target.points: RandomPointTriangle targets are repeatable"""
    tgt1 = points.RandomPointTriangle(
        1776771, [1, 2, 3], [-4.5, 2, 0.7], [1.7, -2, 1], 25, 0.06
    )
    tgt1.prepare()

    tgt2 = points.RandomPointTriangle(
        1776771, [1, 2, 3], [-4.5, 2, 0.7], [1.7, -2, 1], 25, 0.06
    )
    tgt2.prepare()

    # Could probably use == with the position/reflectivity, but I don't trust floats.
    assert len(tgt1) == len(tgt2)
    assert np.allclose(tgt1.position, tgt2.position)
    assert np.allclose(tgt1.reflectivity, tgt2.reflectivity)


@pytest.mark.parametrize(
    "position,reflectivity",
    [
        ([0, 0, 0], 1),
        ([10.5, 77.1, 8.8], 0.01),
    ],
)
def test_target_points_single(position, reflectivity):
    """target.points: SinglePoint"""
    tgt = points.SinglePoint(position, reflectivity)
    tgt.prepare()
    assert len(tgt) == 1
    assert tgt.position.shape == (1, 3)
    assert tgt.reflectivity.shape == (1,)
    assert np.allclose(tgt.position, position)
    assert np.allclose(tgt.reflectivity, reflectivity)

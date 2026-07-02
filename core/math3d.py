"""
math3d.py

Core 3D vector mathematics used throughout KS-Optics.

This module intentionally depends only on NumPy.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ==========================================================
# Basic vector operations
# ==========================================================

def length(v: np.ndarray) -> float:
    """Return the Euclidean norm of a vector."""
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Return a normalized copy of a vector.

    Raises
    ------
    ValueError
        If the vector has zero length.
    """
    v = np.asarray(v, dtype=float)

    n = np.linalg.norm(v)

    if n < EPS:
        raise ValueError("Cannot normalize a zero-length vector.")

    return v / n


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product."""
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product."""
    return np.cross(a, b)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return length(np.asarray(a) - np.asarray(b))


# ==========================================================
# Geometry
# ==========================================================

def angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the angle (radians) between two vectors.
    """
    a = normalize(a)
    b = normalize(b)

    c = np.clip(dot(a, b), -1.0, 1.0)

    return float(np.arccos(c))


def project(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Projection of a vector onto an axis.
    """
    axis = normalize(axis)

    return dot(vector, axis) * axis


def reject(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Component perpendicular to an axis.
    """
    return vector - project(vector, axis)


# ==========================================================
# Optics
# ==========================================================

def reflect(incident: np.ndarray,
            normal: np.ndarray) -> np.ndarray:
    """
    Reflect a unit vector about a surface normal.

    Parameters
    ----------
    incident
        Incoming ray direction.

    normal
        Outward unit normal.

    Returns
    -------
    Reflected unit vector.
    """

    incident = normalize(incident)
    normal = normalize(normal)

    reflected = incident - 2.0 * dot(incident, normal) * normal

    return normalize(reflected)


# ==========================================================
# Coordinate Frames
# ==========================================================

def build_basis(direction: np.ndarray):
    """
    Construct a right-handed orthonormal basis.

    Returns
    -------
    ex, ey, ez
    """

    ez = normalize(direction)

    if abs(ez[2]) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])
    else:
        helper = np.array([0.0, 0.0, 1.0])

    ex = normalize(cross(helper, ez))
    ey = normalize(cross(ez, ex))

    return ex, ey, ez


# ==========================================================
# Coordinate transforms
# ==========================================================

def local_to_global(origin,
                    ex,
                    ey,
                    ez,
                    point):
    """
    Convert a local point to global coordinates.
    """

    return (
        origin
        + point[0] * ex
        + point[1] * ey
        + point[2] * ez
    )


def global_to_local(origin,
                    ex,
                    ey,
                    ez,
                    point):
    """
    Convert a global point into local coordinates.
    """

    p = point - origin

    return np.array([
        dot(p, ex),
        dot(p, ey),
        dot(p, ez)
    ])
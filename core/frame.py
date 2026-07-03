"""
frame.py

Local Cartesian coordinate frame.

Every optical object in the simulator owns one Frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .math3d import (
    normalize,
    dot,
    cross,
    local_to_global,
    global_to_local
)


@dataclass
class Frame:
    """
    Right-handed orthonormal coordinate frame.

    Parameters
    ----------
    origin : ndarray
        Frame origin.

    ex, ey, ez : ndarray
        Local orthonormal basis vectors.
    """

    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray

    def __post_init__(self):

        self.origin = np.asarray(self.origin, dtype=float)

        self.ex = normalize(self.ex)
        self.ey = normalize(self.ey)
        self.ez = normalize(self.ez)

        # Re-orthogonalize to reduce numerical drift
        self.ez = normalize(cross(self.ex, self.ey))
        self.ey = normalize(cross(self.ez, self.ex))

    # -------------------------------------------------

    @classmethod
    def from_z_axis(cls,
                    origin,
                    direction):
        """
        Construct a frame from a propagation direction.
        """

        ez = normalize(direction)

        if abs(ez[2]) > 0.99:
            helper = np.array([0., 1., 0.])
        else:
            helper = np.array([0., 0., 1.])

        ex = normalize(cross(helper, ez))
        ey = normalize(cross(ez, ex))

        return cls(origin, ex, ey, ez)

    # -------------------------------------------------

    def to_global(self,
                  point_local):
        """
        Local → Global coordinates.
        """

        return local_to_global(
            self.origin,
            self.ex,
            self.ey,
            self.ez,
            point_local
        )

    # -------------------------------------------------

    def to_local(self,
                 point_global):
        """
        Global → Local coordinates.
        """

        return global_to_local(
            self.origin,
            self.ex,
            self.ey,
            self.ez,
            point_global
        )

    # -------------------------------------------------

    def translate(self,
                  displacement):
        """
        Translate the frame.
        """

        self.origin += np.asarray(displacement)

    # -------------------------------------------------

    def copy(self):

        return Frame(
            self.origin.copy(),
            self.ex.copy(),
            self.ey.copy(),
            self.ez.copy()
        )

    # -------------------------------------------------

    @property
    def rotation_matrix(self):
        """
        Local → Global rotation matrix.
        """

        return np.column_stack(
            (
                self.ex,
                self.ey,
                self.ez
            )
        )

    # -------------------------------------------------

    def __repr__(self):

        return (
            "Frame(\n"
            f" origin={self.origin},\n"
            f" ex={self.ex},\n"
            f" ey={self.ey},\n"
            f" ez={self.ez}\n"
            ")"
        )
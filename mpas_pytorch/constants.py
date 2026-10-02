"""
Physical constants for MPAS-Atmosphere dynamics.

This module defines the physical constants used by the dynamics core. Double
precision is the model's canonical numerical precision.

Upstream MPAS variable names:
    pii     -> PI
    a       -> EARTH_RADIUS
    omega   -> OMEGA
    gravity -> G
    rgas    -> RD
    rv      -> RV
    cp      -> CP
    cv      -> CV
    rvord   -> RVORD
    cvpm    -> CVPM
    p0      -> P0
    prandtl -> PRANDTL
"""

from typing import ClassVar

import torch

# Default dtype for all computations (double precision)
DTYPE = torch.float64


class _ImmutableConstantsMeta(type):
    """Metaclass that prevents modification of class attributes."""

    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(
                f"Cannot modify PhysicalConstants.{name}: constants are immutable"
            )
        # Allow setting new attributes during class definition only
        super().__setattr__(name, value)

    def __delattr__(cls, name):
        raise AttributeError(
            f"Cannot delete PhysicalConstants.{name}: constants are immutable"
        )


class PhysicalConstants(metaclass=_ImmutableConstantsMeta):
    """Container for MPAS physical constants.

    IMMUTABLE: All constants are read-only. Any attempt to modify them will
    raise AttributeError.

    All constants are stored as Python float literals. PyTorch kernels
    automatically promote these to tensor scalars on the active device.
    Values follow the upstream MPAS registry and parameter definitions.
    """

    # Mathematical constant Pi
    # Fortran name: pii
    PI = 3.141592653589793

    # Gravitational acceleration (m/s^2)
    # Fortran name: gravity
    G = 9.80616

    # Gas constant for dry air (J/(kg*K))
    # Fortran name: rgas
    RD = 287.0

    # Gas constant for water vapor (J/(kg*K))
    # Fortran name: rv
    RV = 461.6

    # Specific heat at constant pressure (J/(kg*K))
    # Fortran name: cp
    CP = 1004.5

    # Specific heat at constant volume (J/(kg*K))
    # Fortran name: cv
    CV = 717.5

    # Earth's mean radius (m)
    # Fortran name: a
    EARTH_RADIUS = 6371229.0

    # Earth's rotational angular velocity (rad/s)
    # Fortran name: omega
    OMEGA = 7.29212e-5

    # Standard atmospheric pressure (Pa)
    # Fortran name: p0
    P0 = 100000.0

    # Ratio of gas constants: rv / rgas
    # Fortran name: rvord
    RVORD = 1.6083623693379792

    # Moist air specific heat adjustment: -cv/cp (negative ratio)
    # Fortran name: cvpm
    CVPM = -0.7142857142857143

    # Prandtl number (dimensionless)
    # Upstream MPAS name: prandtl
    PRANDTL = 1.0

    # Reference temperature (K)
    # Auxiliary thermodynamic constant retained for compatibility.
    T0 = 288.15

    # Latent heat of vaporization (J/kg)
    # Auxiliary thermodynamic constant retained for compatibility.
    LVP = 2.5e6

    # Latent heat of fusion (J/kg)
    # Auxiliary thermodynamic constant retained for compatibility.
    LF = 3.34e5

    # Mapping from upstream MPAS names to Python attribute names.
    UPSTREAM_NAME_MAP: ClassVar[dict[str, str]] = {
        "pii": "PI",
        "a": "EARTH_RADIUS",
        "omega": "OMEGA",
        "gravity": "G",
        "rgas": "RD",
        "rv": "RV",
        "cp": "CP",
        "cv": "CV",
        "rvord": "RVORD",
        "cvpm": "CVPM",
        "p0": "P0",
        "prandtl": "PRANDTL",
    }

    def __setattr__(self, name, value):
        """Prevent instance attribute modification."""
        raise AttributeError(
            f"Cannot modify PhysicalConstants.{name}: constants are immutable"
        )

    def __delattr__(self, name):
        """Prevent instance attribute deletion."""
        raise AttributeError(
            f"Cannot delete PhysicalConstants.{name}: constants are immutable"
        )

    @classmethod
    def get_dict(cls) -> dict[str, float]:
        """Return all constants as a dictionary."""
        return {
            "g": cls.G,
            "rd": cls.RD,
            "rv": cls.RV,
            "cp": cls.CP,
            "cv": cls.CV,
            "earth_radius": cls.EARTH_RADIUS,
            "omega": cls.OMEGA,
            "p0": cls.P0,
            "t0": cls.T0,
            "lvp": cls.LVP,
            "lf": cls.LF,
            "pi": cls.PI,
            "rvord": cls.RVORD,
            "cvpm": cls.CVPM,
            "prandtl": cls.PRANDTL,
        }

    @classmethod
    def get_upstream_dict(cls) -> dict[str, float]:
        """Return constants using upstream MPAS variable names."""
        return {
            upstream_name: getattr(cls, python_name)
            for upstream_name, python_name in cls.UPSTREAM_NAME_MAP.items()
        }


# Module-level exports for convenience
# (These are float values extracted from tensors for easier arithmetic)
GRAVITY = 9.80616
OMEGA = 7.29212e-5
RGAS = 287.0
CP = 1004.5
CV = 717.5
P0 = 100000.0
PRANDTL = 1.0
EARTH_RADIUS = 6371229.0
PI = 3.141592653589793

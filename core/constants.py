"""
constants.py

Physical constants used throughout KS-Optics.

All values are in SI units.
"""

import numpy as np

# ==========================================================
# Mathematical Constants
# ==========================================================

PI = np.pi
TWO_PI = 2.0 * np.pi

# ==========================================================
# Fundamental Physical Constants
# ==========================================================

# Speed of light in vacuum (m/s)
C = 299_792_458.0

# Vacuum permeability (H/m)
MU0 = 4e-7 * np.pi

# Vacuum permittivity (F/m)
EPS0 = 1.0 / (MU0 * C**2)

# Free-space wave impedance (Ohms)
ETA0 = np.sqrt(MU0 / EPS0)

# ==========================================================
# Default Gyrotron Parameters
# ==========================================================

# Default operating frequency (Hz)
DEFAULT_FREQUENCY = 140e9

# Default Gaussian beam waist radius (m)
DEFAULT_WAIST_RADIUS = 0.020

# Default transmitted power (W)
DEFAULT_POWER = 1.0
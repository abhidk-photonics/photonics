import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1
from scipy.interpolate import CubicSpline

C0 = 299792458.0
# Sellmeier coefficients for fused silica (Malitson, 1965), wavelength in µm.
B = np.array([0.6961663, 0.4079426, 0.8974794])
C = np.array([0.004679148, 0.01351206, 97.934002])

def silica_index(wavelength_um):
    """Malitson Sellmeier refractive index for fused silica; wavelength in µm."""
    x = np.asarray(wavelength_um, dtype=float)
    x2 = x**2
    n2 = 1 + sum(B[i] * x2 / (x2 - C[i]) for i in range(3))
    return np.sqrt(n2)

def characteristic_equation(u, V):
    """LP01 scalar characteristic equation, expressed for root finding."""
    # NaN outside (0, V) because w = sqrt(V^2 - u^2) must be real and positive.
    if u <= 0 or u >= V:
        return np.nan
    w = np.sqrt(V**2 - u**2)
    return u*j1(u)/j0(u) - w*k1(w)/k0(w)

def solve_lp01_b(V):
    """Find normalized propagation parameter b for the fundamental LP01 mode.

    The fundamental root lies below the first zero of J0, so a direct bracket
    avoids scanning a dense grid for every wavelength/design combination.
    """
    # Stay slightly inside the interval: the equation is singular at u = 0 and
    # at u = V (K0 diverges when w -> 0) and at the zero of J0.
    eps = max(1e-10, V*1e-8)
    upper = min(V, 2.404825557695773) - eps  # first positive zero of J0
    f_lo = characteristic_equation(eps, V)
    f_hi = characteristic_equation(upper, V)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo*f_hi >= 0:
        raise RuntimeError(f"Could not bracket LP01 root for V={V}")
    u = brentq(characteristic_equation, eps, upper, args=(V,), xtol=1e-12)
    return 1 - (u/V)**2

def calculate_neff(core_radius_um, delta_n, wavelength_nm):
    """Effective index of the fundamental LP01 mode."""
    lam_um = np.asarray(wavelength_nm, dtype=float)/1000
    ncl = silica_index(lam_um)
    # Simplification: the core-cladding contrast is a constant delta_n, so only
    # the cladding carries the Sellmeier wavelength dependence.
    nco = ncl + delta_n
    V = 2*np.pi*core_radius_um/lam_um * np.sqrt(nco**2 - ncl**2)
    b = np.array([solve_lp01_b(float(v)) for v in np.atleast_1d(V)])
    if np.ndim(V) == 0:
        b = b[0]
    result = np.sqrt(ncl**2 + b*(nco**2 - ncl**2))
    return float(result) if np.ndim(result) == 0 else result

def calculate_dispersion_curve(core_radius_um, delta_n, wavelength_nm):
    """Return D in ps/(nm·km), using a cubic spline of neff versus wavelength."""
    lam_m = np.asarray(wavelength_nm, dtype=float)*1e-9
    neff = np.asarray(calculate_neff(core_radius_um, delta_n, wavelength_nm))
    # The second derivative is taken analytically from the spline. It is least
    # reliable at the two ends of the wavelength grid.
    spline = CubicSpline(lam_m, neff)
    d2n = spline(lam_m, 2)
    # D = -(lambda/c) d2n/dlambda2 in s/m^2; 1 s/m^2 = 1e6 ps/(nm·km).
    return (-(lam_m/C0)*d2n)*1e6

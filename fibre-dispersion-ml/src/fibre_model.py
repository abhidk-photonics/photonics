import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1


C0 = 299792458.0

# Malitson (1965) Sellmeier coefficients for fused silica. Wavelength is in µm.
B = np.array([0.6961663, 0.4079426, 0.8974794])
C = np.array([0.004679148, 0.01351206, 97.934002])


def silica_index(wavelength_um):
    """Evaluate the fused-silica refractive index using the Malitson Sellmeier relation."""
    wavelength = np.asarray(wavelength_um, dtype=float)
    wavelength_sq = wavelength**2
    n_squared = 1 + sum(
        B[i] * wavelength_sq / (wavelength_sq - C[i])
        for i in range(3)
    )
    return np.sqrt(n_squared)


def characteristic_equation(u, V):
    """Evaluate the scalar LP01 characteristic equation for root finding."""
    # Restrict the domain so the cladding decay parameter remains real and positive.
    if u <= 0 or u >= V:
        return np.nan

    w = np.sqrt(V**2 - u**2)
    return u * j1(u) / j0(u) - w * k1(w) / k0(w)


def solve_lp01_b(V):
    """Solve for the normalised propagation parameter of the fundamental LP01 mode."""
    # Bracket the fundamental root below the first positive zero of J0.
    eps = max(1e-10, V * 1e-8)
    upper = min(V, 2.404825557695773) - eps
    f_lower = characteristic_equation(eps, V)
    f_upper = characteristic_equation(upper, V)

    if not np.isfinite(f_lower) or not np.isfinite(f_upper) or f_lower * f_upper >= 0:
        raise RuntimeError(f"Could not bracket LP01 root for V={V}")

    u = brentq(
        characteristic_equation,
        eps,
        upper,
        args=(V,),
        xtol=1e-12,
    )
    return 1 - (u / V) ** 2


def calculate_neff(core_radius_um, delta_n, wavelength_nm):
    """Calculate the effective index of the fundamental LP01 mode."""
    wavelength_um = np.asarray(wavelength_nm, dtype=float) / 1000
    n_clad = silica_index(wavelength_um)

    # The index contrast is fixed, so Sellmeier dispersion enters through the cladding.
    n_core = n_clad + delta_n
    V = (
        2 * np.pi * core_radius_um / wavelength_um
        * np.sqrt(n_core**2 - n_clad**2)
    )
    b = np.array([solve_lp01_b(float(value)) for value in np.atleast_1d(V)])

    if np.ndim(V) == 0:
        b = b[0]

    result = np.sqrt(n_clad**2 + b * (n_core**2 - n_clad**2))
    return float(result) if np.ndim(result) == 0 else result


def calculate_dispersion_curve(core_radius_um, delta_n, wavelength_nm):
    """Calculate chromatic dispersion in ps/(nm·km) using spline differentiation."""
    wavelength_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    neff = np.asarray(
        calculate_neff(core_radius_um, delta_n, wavelength_nm)
    )

    # The spline gives the second wavelength derivative. Endpoints are less reliable.
    spline = CubicSpline(wavelength_m, neff)
    d2n = spline(wavelength_m, 2)

    # Convert D = -(λ/c)d²n_eff/dλ² from s/m² to ps/(nm·km).
    return (-(wavelength_m / C0) * d2n) * 1e6

"""Finite-strain 2D elastic constants from an ASE calculator. Runs in the MLIP environment (torch, ase).

Protocol (mirrors the C2DB asr.stiffness files: +/-1% strains in xx, yy, xy; ions relaxed at each strained cell):
  1. Orient the cell: a along x, b in the xy plane, c (vacuum direction) along z.
  2. Optional reference relaxation (variant "relaxed_cell"): ions + in-plane cell (xx, yy, xy), c fixed.
     Variant "dft_cell" skips the cell relaxation and keeps the database geometry (ions are still relaxed).
  3. For each strain in {xx, yy, xy} x {-d, +d}: deform the cell, relax ions only, read the stress.
  4. Central differences: C_ij = (sigma_i(+d) - sigma_i(-d)) / (2 d), then 2D: C[N/m] = C[eV/A^3] * Lz * EV_A2_TO_N_M.

Conventions that matter (all checked numerically in run_calibration.py, not assumed):
  * ASE stress is positive in tension, Voigt order (xx, yy, zz, yz, xz, xy), unit eV/A^3.
  * Lz = cell volume / in-plane area (full cell height, vacuum included). 2D stress = 3D stress * Lz.
  * xy strain is the ENGINEERING shear strain gamma_xy = 2*eps_xy (Voigt), applied as a simple shear x -> x + gamma*y.
    The xy constant is therefore C66 in Voigt notation. C2DB's convention for c_33 is unverified, so xy results are
    diagnostic only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS

EV_A2_TO_N_M = 16.02176634  # 1 eV/A^2 = 16.0218 J/m^2


@dataclass
class Settings:
    strain: float = 0.01
    fmax: float = 0.005
    max_steps: int = 600
    optimizer: str = "LBFGS"
    variant: str = "relaxed_cell"          # relaxed_cell | dft_cell
    cell_mask: tuple = (1, 1, 0, 0, 0, 1)  # xx, yy, zz, yz, xz, xy : relax in-plane cell only
    meta: dict = field(default_factory=dict)


MIN_VACUUM = 15.0  # A of empty space between periodic images along c; must exceed every model cutoff (MACE 6 A, CHGNet 5 A)


def orient(atoms: Atoms, min_vacuum: float = MIN_VACUUM) -> Atoms:
    """Standard orientation (a || x, b in xy), vacuum axis c || z, slab centred, vacuum gap >= min_vacuum.

    The cell is made PERIODIC IN ALL THREE DIRECTIONS. With pbc_z=False the MACE ASE calculator returns a stress that is
    wrong by a constant factor (measured 0.185 for a 34.9 A cell; with pbc_z=True the analytic/finite-difference ratio is
    1.0000002). 2D quantities do not depend on the vacuum size because the 3D stress is multiplied by the cell height.

    Cells whose c axis is tilted from the plane normal (present in JARVIS) are handled by re-describing the slab with
    c = (0, 0, L): this is exact for a slab that is contiguous in the given Cartesian coordinates, because the tilt only
    changes how the periodic images are stacked in the vacuum. Slabs that straddle the cell boundary in a tilted cell are
    rejected (wrapping by a tilted c would change the in-plane structure).
    """
    a, b, c = atoms.cell.array
    ex = a / np.linalg.norm(a)
    ez = np.cross(a, b)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    pos = atoms.get_positions() @ np.array([ex, ey, ez]).T          # components along ex, ey, ez
    h = abs(float(c @ ez))
    tilt = float(np.linalg.norm(c - (c @ ez) * ez))
    z = pos[:, 2]
    if tilt < 1e-6 * np.linalg.norm(c):
        # perpendicular c: wrapping along c keeps the in-plane position, so slabs straddling the boundary are fine
        zw = z % h
        zs = np.sort(zw)
        gaps = np.diff(np.r_[zs, zs[0] + h])
        k = int(np.argmax(gaps))
        start = zs[(k + 1) % len(zs)]
        zrel = (zw - start) % h
    else:
        zrel = z - z.min()
        if h - float(zrel.max()) < 5.0:
            raise ValueError(f"tilted cell with a slab that may straddle the boundary (extent {zrel.max():.2f} of height {h:.2f})")
    extent = float(zrel.max())
    newL = max(h, extent + min_vacuum)
    new_pos = np.column_stack([pos[:, 0], pos[:, 1], zrel + (newL - extent) / 2])
    cell = np.array([[np.linalg.norm(a), 0, 0], [float(b @ ex), float(b @ ey), 0], [0, 0, newL]])
    return Atoms(atoms.numbers, cell=cell, positions=new_pos, pbc=True)


def lz(atoms: Atoms) -> float:
    return atoms.get_volume() / np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))


def _optimizer(name, atoms):
    return {"LBFGS": LBFGS, "FIRE": FIRE}[name](atoms, logfile=None)


def relax(atoms: Atoms, s: Settings, cell: bool):
    obj = FrechetCellFilter(atoms, mask=list(s.cell_mask)) if cell else atoms
    opt = _optimizer(s.optimizer, obj)
    conv = opt.run(fmax=s.fmax, steps=s.max_steps)
    f = obj.get_forces()
    return {"converged": bool(conv), "steps": int(opt.get_number_of_steps()), "fmax": float(np.sqrt((f**2).sum(axis=1)).max())}


def strained(atoms: Atoms, voigt: int, d: float) -> Atoms:
    """Apply strain d (engineering for xy) along Voigt index 0 (xx), 1 (yy) or 5 (xy)."""
    eps = np.eye(3)
    if voigt == 0:
        eps[0, 0] += d
    elif voigt == 1:
        eps[1, 1] += d
    elif voigt == 5:
        eps[0, 1] += d  # row-vector convention: new_cell = cell @ eps.T ; adds d to x per unit y => gamma_xy = d
    else:
        raise ValueError(voigt)
    a = atoms.copy()
    a.set_cell(atoms.cell.array @ eps.T, scale_atoms=True)
    return a


def stress_voigt(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.get_stress(), float)  # Voigt (xx,yy,zz,yz,xz,xy), eV/A^3


def elastic_constants(atoms0: Atoms, calc, s: Settings) -> dict:
    """Return the 2D elastic constants (N/m) and diagnostics. atoms0 must already be oriented."""
    t0 = time.time()
    a = atoms0.copy()
    a.calc = calc
    info = {"variant": s.variant, "n_atoms": len(a)}
    ref_stress = None
    if s.variant == "relaxed_cell":
        info["ref_relax"] = relax(a, s, cell=True)
    elif s.variant == "dft_cell":
        info["ref_relax"] = relax(a, s, cell=False)
    else:
        raise ValueError(s.variant)
    info["a_b_gamma"] = a.cell.cellpar()[[0, 1, 5]].tolist()
    info["Lz"] = lz(a)
    ref_stress = stress_voigt(a)
    info["ref_stress_xx_yy_xy_eV_A3"] = ref_stress[[0, 1, 5]].tolist()

    sig = {}
    conv_all, steps = True, 0
    for v in (0, 1, 5):
        for sign in (-1, 1):
            b = strained(a, v, sign * s.strain)
            b.calc = calc
            r = relax(b, s, cell=False)
            conv_all &= r["converged"]
            steps += r["steps"]
            sig[(v, sign)] = stress_voigt(b)
    C = np.zeros((3, 3))
    idx = [0, 1, 5]
    for j, v in enumerate(idx):
        d = (sig[(v, 1)] - sig[(v, -1)]) / (2 * s.strain)   # eV/A^3 per unit strain, all Voigt components
        for i, vi in enumerate(idx):
            C[i, j] = d[vi]
    C *= info["Lz"] * EV_A2_TO_N_M
    # symmetrise for derived quantities, keep the raw asymmetry as a diagnostic
    asym = float(np.abs(C - C.T).max() / max(np.abs(C).max(), 1e-12))
    Cs = (C + C.T) / 2
    w = np.linalg.eigvalsh(Cs)
    c11, c22, c12, c66 = Cs[0, 0], Cs[1, 1], Cs[0, 1], Cs[2, 2]
    info.update({"c11": c11, "c22": c22, "c12": c12, "c66": c66, "c11_raw": C[0, 0], "c22_raw": C[1, 1],
                 "c12_raw": C[0, 1], "c21_raw": C[1, 0], "asym_frac": asym, "min_eig": float(w[0]),
                 "stable_tensor": bool(w[0] > 0), "strain_relax_converged": bool(conv_all), "strain_relax_steps": steps,
                 "Y2D": (c11 * c22 - c12**2) / c22 if c22 != 0 else np.nan, "poisson": c12 / c22 if c22 != 0 else np.nan,
                 "seconds": time.time() - t0})
    return info


def energy_stress_check(atoms0: Atoms, calc, d: float = 1e-3, pre_strain: float = 0.02) -> dict:  # d=1e-4 is too small for float32 models (CHGNet): FD noise up to +-7%; 1e-3 gives 1.000 +- 0.004
    """Implementation check: analytic stress vs finite-difference of the energy under strain (no relaxation).

    sigma_xx(analytic) should equal (1/V) dE/d(eps_xx), sigma_xy(analytic) should equal (1/V) dE/d(gamma_xy).
    Returns both and their ratio; a ratio far from 1 flags a sign/unit/convention error in the stress path.
    """
    # stretched and sheared so every checked stress component is far above numerical noise
    a = strained(strained(strained(atoms0, 0, pre_strain), 1, 0.5 * pre_strain), 5, pre_strain)
    a.calc = calc
    sig = a.get_stress()
    V = a.get_volume()
    out = {}
    for name, v in (("xx", 0), ("yy", 1), ("xy", 5)):
        ep, em = strained(a, v, d), strained(a, v, -d)
        ep.calc, em.calc = calc, calc
        fd = (ep.get_potential_energy() - em.get_potential_energy()) / (2 * d) / V
        out[name] = {"analytic": float(sig[v]), "finite_diff": float(fd),
                     "ratio": float(sig[v] / fd) if abs(fd) > 1e-8 else None}
    return out

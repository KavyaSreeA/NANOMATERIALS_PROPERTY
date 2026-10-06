"""Read a CIF / POSCAR of a vacuum-padded monolayer and derive the geometry inputs the model needs (same definitions as mat-ml/src/data.py)."""
from __future__ import annotations

import io

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def slab_geometry(lattice: np.ndarray, frac: np.ndarray) -> dict:
    """In-plane lattice, area and slab thickness along the surface normal (copy of data.slab_geometry)."""
    a, b, c = lattice
    cross = np.cross(a, b)
    area = float(np.linalg.norm(cross))
    n = cross / area
    height = abs(float(np.dot(c, n)))
    z = np.sort(((frac @ lattice) @ n) % height)
    gaps = np.diff(np.r_[z, z[0] + height])
    cosg = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return {"a": float(np.linalg.norm(a)), "b": float(np.linalg.norm(b)), "gamma": float(np.degrees(np.arccos(np.clip(cosg, -1, 1)))),
            "area": area, "cell_height": height, "z_extent": float(height - gaps.max())}


def read_structure(content: bytes, filename: str) -> Structure:
    text = content.decode("utf-8", errors="replace")
    name = (filename or "").lower()
    if name.endswith(".cif"):
        fmt = "cif"
    elif name.endswith((".json",)):
        fmt = "json"
    else:
        fmt = "poscar"
    try:
        return Structure.from_str(text, fmt=fmt)
    except Exception as exc:
        raise ValueError(f"could not read '{filename}' as a {fmt.upper()} structure: {exc}") from exc


def describe(structure: Structure, symprec: float = 0.01) -> tuple[dict, list[str]]:
    """Return (inputs for the model, warnings)."""
    warnings: list[str] = []
    geo = slab_geometry(structure.lattice.matrix, structure.frac_coords)
    nat = len(structure)
    if nat > 200:
        raise ValueError("structure has more than 200 atoms; this model was trained on small primitive monolayer cells")
    try:
        spg = int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        spg = 1
        warnings.append("space group could not be determined; 1 was used")
    vacuum = geo["cell_height"] - geo["z_extent"]
    if vacuum < 8.0:
        warnings.append(f"only {vacuum:.1f} Å of vacuum along the out-of-plane direction: this does not look like an isolated monolayer, so the prediction may be meaningless")
    if abs(structure.lattice.alpha - 90) > 5 or abs(structure.lattice.beta - 90) > 5:
        warnings.append("the c axis is not perpendicular to the a-b plane; the first two lattice vectors are assumed to span the sheet")
    warnings.append("the space-group number is computed on the slab with symprec=0.01; the training data used C2DB's own value, which may differ slightly")
    inputs = {"formula": structure.composition.reduced_formula, "a": geo["a"], "b": geo["b"], "gamma": geo["gamma"], "nat": nat,
              "thickness": geo["z_extent"], "spg_number": spg, "area_per_atom": geo["area"] / nat}
    return inputs, warnings

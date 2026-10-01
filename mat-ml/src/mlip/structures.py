"""Build ASE Atoms from the benchmark JSON items (runs in the MLIP environment)."""
from ase import Atoms

from .stiffness import orient


def load_item(item: dict) -> Atoms:
    kw = {"numbers": item["numbers"]} if item.get("numbers") else {"symbols": item["symbols"]}
    return orient(Atoms(cell=item["cell"], positions=item["positions"], pbc=item["pbc"], **kw))

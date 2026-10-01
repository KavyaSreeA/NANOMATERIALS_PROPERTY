"""Calculator factory for the universal potentials. Runs in the MLIP environment.

Verified against the installed packages (mace-torch 0.3.16, chgnet 0.4.2), not from memory:
  mace_mp(model=<name|path>, device, default_dtype, dispersion, ...)  names include 'medium', 'medium-mpa-0', ...
  CHGNetCalculator(model=None -> pretrained, use_device=..., stress_weight=1/160.21)  (calculator converts GPa -> eV/A^3)
Models under the Academic Software License (the *-omat-* and matpes checkpoints) are deliberately not used here.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

MODELS = {
    "mace_mp0_medium": ("mace", "medium"),
    "mace_mpa0_medium": ("mace", "medium-mpa-0"),
    "chgnet_0.3.0": ("chgnet", "0.3.0"),
}


def build(name: str, device: str = "cuda", dtype: str = "float64", dispersion: bool = False):
    kind, tag = MODELS[name]
    if kind == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model=tag, device=device, default_dtype=dtype, dispersion=dispersion)
    if kind == "chgnet":
        if dispersion:
            raise NotImplementedError("dispersion not wired for CHGNet in this pipeline")
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(model=CHGNet.load(model_name=tag, use_device=device, verbose=False), use_device=device)
    raise ValueError(name)


def versions() -> dict:
    import ase
    import chgnet
    import mace
    import torch
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "ase": ase.__version__,
            "mace": getattr(mace, "__version__", "?"), "chgnet": getattr(chgnet, "__version__", "?")}

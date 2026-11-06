"""Optional RDKit helpers (SMILES → 3-D). Everything degrades gracefully when
RDKit is not installed."""
from __future__ import annotations

import numpy as np

from .molio import Molecule, make_molecule


def rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def rdkit_version() -> str:
    try:
        import rdkit
        return rdkit.__version__
    except Exception:
        return "not installed"


def smiles_to_molecule(smiles: str, seed: int = 42, optimize: str = "MMFF94", max_iters: int = 2000,
                       title: str = "single_chain") -> tuple[Molecule, dict]:
    """Embed a SMILES string in 3-D with ETKDGv3 and optionally optimise it."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        raise ValueError("RDKit could not parse this SMILES string.")
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    ps.useRandomCoords = mh.GetNumAtoms() > 120
    cid = AllChem.EmbedMolecule(mh, ps)
    if cid < 0:
        ps.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mh, ps)
    if cid < 0:
        raise RuntimeError("3-D embedding failed – try a shorter chain or another random seed.")
    info = {"engine": "RDKit ETKDGv3", "seed": seed, "optimiser": optimize, "converged": None}
    if optimize == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mh):
        res = AllChem.MMFFOptimizeMolecule(mh, maxIters=max_iters)
        info["converged"] = res == 0
    elif optimize in ("UFF", "MMFF94"):
        res = AllChem.UFFOptimizeMolecule(mh, maxIters=max_iters)
        info["converged"] = res == 0
        info["optimiser"] = "UFF"
    info["formula"] = rdMolDescriptors.CalcMolFormula(mh)
    info["exact MW"] = round(Descriptors.ExactMolWt(mh), 3)
    info["stereocentres"] = len(Chem.FindMolChiralCenters(mh, includeUnassigned=True))
    pos = mh.GetConformer().GetPositions()
    els = [a.GetSymbol().upper() for a in mh.GetAtoms()]
    q = [a.GetFormalCharge() for a in mh.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mh.GetBonds()]
    orders = [4 if b.GetIsAromatic() else int(round(b.GetBondTypeAsDouble())) for b in mh.GetBonds()]
    mol = make_molecule(els, pos, charges=q, bonds=bonds, orders=orders, title=title)
    return mol, info


def molecule_to_smiles(mol: Molecule) -> str:
    """Canonical isomeric SMILES with stereochemistry perceived from 3-D coordinates."""
    from rdkit import Chem
    from .forcefield import to_rdkit
    rm = to_rdkit(mol, add_h=False)
    Chem.AssignStereochemistryFrom3D(rm)
    return Chem.MolToSmiles(Chem.RemoveHs(rm))


EXAMPLE_SMILES = {
    "Alginate trimer G-G-M (stereo-free example)":
        "OC1OC(C(=O)O)C(OC2OC(C(=O)O)C(OC3OC(C(=O)O)C(O)C(O)C3O)C(O)C2O)C(O)C1O",
    "Polystyrene trimer": "CC(c1ccccc1)CC(c1ccccc1)CC(C)c1ccccc1",
    "Polyethylene C20": "CCCCCCCCCCCCCCCCCCCC",
}

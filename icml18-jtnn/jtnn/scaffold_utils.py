from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Set

import rdkit.Chem as Chem


@dataclass
class ScaffoldContext:
    scaffold_mol: Chem.Mol
    scaffold_source: Literal["smiles", "smarts"]
    strict: bool = True
    final_substruct_check: bool = True
    allowed_root_wids: Optional[Set[int]] = None
    allowed_wids: Optional[Set[int]] = None
    max_decode_retry: int = 8
    anchor_atom_idx: Optional[int] = None
    unique_anchor_only: bool = False
    require_anchor_growth: bool = True
    meta: Optional[Dict[str, Any]] = None


def filter_wids_by_scaffold(vocab, scaffold_smiles: str) -> Set[int]:
    """
    Derive scaffold clique ids from MolTree(scaffold_smiles), then map them into vocab ids.
    This is used only as a root prior (safe for backward compatibility).
    """
    try:
        from mol_tree import MolTree  # local import to avoid circular imports
    except Exception:
        return set()

    try:
        tree = MolTree(scaffold_smiles)
        clique_smiles = {node.smiles for node in tree.nodes}
    except Exception:
        return set()

    wids = set()
    for sml in clique_smiles:
        try:
            wids.add(vocab.get_index(sml))
        except Exception:
            continue
    return wids


def build_scaffold_context(
    vocab,
    scaffold_smiles: Optional[str] = None,
    scaffold_smarts: Optional[str] = None,
    strict: bool = True,
    final_substruct_check: bool = True,
    build_vocab_masks: bool = True,
    max_decode_retry: int = 8,
    anchor_atom_idx: Optional[int] = None,
    unique_anchor_only: bool = False,
    require_anchor_growth: bool = True,
):
    if scaffold_smiles is None and scaffold_smarts is None:
        return None

    if scaffold_smiles is not None and scaffold_smarts is not None:
        raise ValueError("Provide either scaffold_smiles or scaffold_smarts, not both.")

    if scaffold_smiles is not None:
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        source = "smiles"
    else:
        scaffold_mol = Chem.MolFromSmarts(scaffold_smarts)
        source = "smarts"

    if scaffold_mol is None:
        raise ValueError("Invalid scaffold string. Could not parse scaffold molecule.")

    n_atoms = scaffold_mol.GetNumAtoms()
    if anchor_atom_idx is not None and (anchor_atom_idx < 0 or anchor_atom_idx >= n_atoms):
        raise ValueError(
            f"anchor_atom_idx={anchor_atom_idx} is out of range for scaffold atoms [0, {n_atoms - 1}]"
        )

    allowed_root_wids = None
    if build_vocab_masks and source == "smiles":
        root_wids = filter_wids_by_scaffold(vocab, scaffold_smiles)
        if len(root_wids) > 0:
            allowed_root_wids = root_wids

    return ScaffoldContext(
        scaffold_mol=scaffold_mol,
        scaffold_source=source,
        strict=strict,
        final_substruct_check=final_substruct_check,
        allowed_root_wids=allowed_root_wids,
        allowed_wids=None,
        max_decode_retry=max_decode_retry,
        anchor_atom_idx=anchor_atom_idx,
        unique_anchor_only=unique_anchor_only,
        require_anchor_growth=require_anchor_growth,
        meta={},
    )


def _match_anchor_constraints(mol: Chem.Mol, match: tuple, ctx: ScaffoldContext) -> bool:
    if ctx.anchor_atom_idx is None:
        return True

    scaffold_atom_to_mol = list(match)
    scaffold_mol_atom_set = set(scaffold_atom_to_mol)
    anchor_mol_idx = scaffold_atom_to_mol[ctx.anchor_atom_idx]

    anchor_out_degree = 0
    for mol_idx in scaffold_atom_to_mol:
        atom = mol.GetAtomWithIdx(mol_idx)
        out_neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in scaffold_mol_atom_set]
        out_deg = len(out_neighbors)

        if ctx.unique_anchor_only and mol_idx != anchor_mol_idx and out_deg > 0:
            return False
        if mol_idx == anchor_mol_idx:
            anchor_out_degree = out_deg

    if ctx.require_anchor_growth and anchor_out_degree == 0:
        return False
    return True


def smiles_contains_scaffold(smiles: str, scaffold_mol: Chem.Mol) -> bool:
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or scaffold_mol is None:
        return False
    return mol.HasSubstructMatch(scaffold_mol)


def smiles_satisfies_scaffold_context(smiles: str, scaffold_ctx: Optional[ScaffoldContext]) -> bool:
    if scaffold_ctx is None:
        return True
    if smiles is None:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    matches = mol.GetSubstructMatches(scaffold_ctx.scaffold_mol)
    if not matches:
        return False

    for match in matches:
        if _match_anchor_constraints(mol, match, scaffold_ctx):
            return True
    return False

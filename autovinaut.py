#!/usr/bin/env python3
"""
Flexible & Rigid Receptor Docking Pipeline
=================================================

A modular, scriptable pipeline for AutoDock Vina docking with support for:
- Flexible and rigid docking modes.
- Ligand-centric or reference-based docking box definitions.
- Explicit handling of rigid cofactors (e.g., HEME), preserved and merged back.
- Flexible shell selection by radius or explicit residue list.
- Optional freezing of residues near a defined center (e.g., metal atom).
- Seamless integration with Meeko for flexible receptor preparation.
- Multi-seed docking runs and clustering of poses based on RMSD.
- Outputs representative pose per cluster in both PDBQT and PDB formats.

Features
--------
* Ligand translation to box center (from protein COM, haem, or custom atom).
* Box center can be inferred or explicitly set via residue:chain:atom.
* Automatic or manual selection of flexible residues, with optional exclusion by proximity to rigid center.
* Retains non-standard cofactors as rigid PDBQT blocks.
* Multi-run docking and hierarchical clustering of resulting poses.
* Converts best poses to PDB, preserving flex residues.

Requirements
------------
- Python ≥ 3.7
- RDKit
- Open Babel
- Meeko
- PDBFixer
- OpenMM
- SciPy
- NumPy
- BioPython
- AutoDock Vina ≥ 1.2

Recommended usage
-----------------
This script is suitable for:
- High-throughput ligand docking campaigns with flexible regions.
- Enzyme-cofactor systems where rigid residues (e.g., heme) must be preserved.
- Integration into larger modeling or screening pipelines.

"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────
import argparse
import collections
import io
import logging
import shutil
import subprocess
import sys
from pathlib import Path, PurePath
from textwrap import indent
from typing import Iterable, Sequence, Set, Union

# ── third-party ─────────────────────────────────────────────────────────
import numpy as np
from Bio.PDB import (
    NeighborSearch,
    PDBParser,
    PDBIO,
    Select,
    Atom,
    Residue,
)
from openbabel import openbabel
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from scipy.cluster.hierarchy import linkage, fcluster
from vina import Vina

# ---------------------------------------------------------------------------
# CONSTANTS & LOGGING
# ---------------------------------------------------------------------------
STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# SUBPROCESS HELPER
# ---------------------------------------------------------------------------

def run(cmd: list[str] | str, capture: bool = False, **kwargs):
    """Run a command with logging and error forwarding."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    logging.debug("$ %s", printable)
    res = subprocess.run(cmd, text=True, capture_output=capture, check=False, **kwargs)
    if res.returncode:
        logging.error(res.stderr.strip() if res.stderr else "Command failed")
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res

# ---------------------------------------------------------------------------
# PDB / PDBQT UTILS – Open Babel
# ---------------------------------------------------------------------------

def ob_convert(src: Path, dst: Path, fmt_in: str, fmt_out: str, extra: list[str] | None = None):
    extra = extra or []
    run(["obabel", str(src), "-i", fmt_in, "-o", fmt_out, "-O", str(dst), *extra])

def pdb_to_pdbqt(src: Path, dst: Path, assign_charges=True):
    extra = ["--partialcharge", "gasteiger"] if assign_charges else []
    run(["obabel", "-ipdb", str(src), "-opdbqt", "-O", str(dst), *extra])

def centre_of_pdb(pdb: Path) -> np.ndarray:
    conv = openbabel.OBConversion(); conv.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol(); conv.ReadFile(mol, str(pdb))
    coords = np.array([[a.GetX(), a.GetY(), a.GetZ()] for a in openbabel.OBMolAtomIter(mol)])
    return coords.mean(axis=0)

def translate_pdb(in_pdb: Path, new_center: np.ndarray, out_pdb: Path):
    conv = openbabel.OBConversion(); conv.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol(); conv.ReadFile(mol, str(in_pdb))
    old = centre_of_pdb(in_pdb); shift = new_center - old
    for at in openbabel.OBMolAtomIter(mol):
        at.SetVector(at.GetX()+shift[0], at.GetY()+shift[1], at.GetZ()+shift[2])
    conv.WriteFile(mol, str(out_pdb))

# ---------------------------------------------------------------------------
# FLEXIBLE‑RECEPTOR PREP (Meeko)

def fix_pdb_missing_atoms(src: Path, dst: Path):
    """Use PDBFixer to add missing heavy atoms/hydrogens before Meeko."""
    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues(); fixer.findMissingAtoms()
    fixer.addMissingAtoms(); fixer.addMissingHydrogens(pH=7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(dst, "w"), keepIds=True)

# ------------------------------------------------------------
# flex_zone.py – helper routines for flexible-vs-frozen choice
# ------------------------------------------------------------

# Old function, not used anymore
def find_flexible_residues(struct, centre: np.ndarray, radius: float):
    """Return Residue objects whose any atom lies within *radius* Å of *centre* and is a standard AA."""
    search = NeighborSearch(list(struct.get_atoms()))
    near_atoms = search.search(list(centre), radius)
    return {at.get_parent() for at in near_atoms if at.get_parent().resname.upper() in STD_AA}

def _tag(res):
    ch = res.get_parent().id or ""   # blank chain → ""
    return f"{res.resname}:{ch}{res.id[1]}"

def find_frozen_residues(struct,
                         centre: np.ndarray,
                         freeze_radius: float,
                         verbose: False,
                         flex_set: Set[Residue.Residue]) -> Set[Residue.Residue]:
    """
    Remove from `flex_set` any residue that has at least one atom within
    `freeze_radius` Å of *any* point in `centres`.

    Parameters
    ----------
    struct : Bio.PDB.Structure.Structure
        Whole receptor structure (needed only to build a neighbour search).
    centres : ndarray
        Points to measure distances from (e.g. the Fe atom or all haem atoms).
    freeze_radius : float
        Cut-off distance in Å.
    flex_set : set[Residue]
        Current candidate flexible residues.

    Returns
    -------
    set[Residue]
        `flex_set` with the “frozen” residues removed.
    """
    search = NeighborSearch(list(struct.get_atoms()))
    near_atoms = search.search(list(centre), freeze_radius)
    print(near_atoms, print(list(centre)))
    frozen = {at.get_parent() for at in near_atoms if at.get_parent().resname.upper() in STD_AA}

    # --- optional log -------------------------------------------------
    if verbose and frozen:
        tags = ", ".join(sorted(_tag(r) for r in frozen))
        print(f"[freeze]  {len(frozen)} residues frozen: {tags}")
    print("LEN FLEX AND FROZEN:", len(flex_set), len(frozen))
    return flex_set - frozen

# ---------------------------------------------------------------------------

_parser = PDBParser(QUIET=True)

def _load_first_model(pdb_path: Path):
    """Return the first model (Bio.PDB.Model) from a PDB file."""
    return _parser.get_structure(pdb_path.stem, str(pdb_path))[0]

def _is_target(res, *, resname, chain, resseq):
    """True if this Bio.PDB.Residue matches the (chain, resseq, resname)."""
    res = (
        res.resname.strip() == resname            # HEM
        and res.get_parent().id == chain          # 'A'
        and res.id[1] == int(resseq)              # 999   (res.id = (' ', 999, ' '))
    )
    return res

parser = PDBParser(QUIET=True)

def load_structure(path):
    return parser.get_structure(path.stem, str(path))[0]  # model 0

# -------------------------------------------------------------------------
# strip_residue : write *everything except* the target residue(s)
# -------------------------------------------------------------------------
def strip_residue(src: Path, dst: Path,
                  resname: str, chain: str, resseq: str):
    model = _load_first_model(src)

    class KeepOther(Select):
        def accept_residue(self, res):
            return not _is_target(res, resname=resname,
                                       chain=chain, resseq=resseq)

    pdbio = PDBIO(); pdbio.set_structure(model)
    pdbio.save(str(dst), KeepOther())       # Path → str fixes the .tell() issue
    
# -------------------------------------------------------------------------
# extract_residue : write *only* the target residue(s)
# -------------------------------------------------------------------------

def extract_residue(src: Path,
                    dst: Union[Path, PurePath, TextIO, io.StringIO, None],
                    resname: str, chain: str, resseq: str,
                    verbose: bool = False) -> str:
    """Write the requested residue to *dst* (or just return the text).

    dst : Path-like → file is created/over-written
          open handle → text is written to that handle
          None       → nothing is written
    Returns the PDB text in every case.
    """
    model = _load_first_model(src)

    class KeepOne(Select):
        def accept_residue(self, res):
            return _is_target(res, resname=resname,
                                   chain=chain, resseq=resseq)

    writer = PDBIO()
    writer.set_structure(model)

    buf = io.StringIO()
    writer.save(buf, KeepOne())          # write *once* into memory
    pdb_txt = buf.getvalue()

    # ---------- decide what to do with the text ----------------------
    if isinstance(dst, (Path, PurePath, str)):
        Path(dst).write_text(pdb_txt)
    elif isinstance(dst, io.TextIOBase):
        dst.write(pdb_txt)               # open file or StringIO
    elif dst is not None:
        raise TypeError(
            "dst must be pathlib.Path, str, open handle, or None"
        )

    if verbose:
        print(f"[extract_residue] {len(pdb_txt.splitlines())} lines "
              f"({resname} {chain} {resseq})")

    return pdb_txt

def parse_center(s):
    """
    Turn 
    "RES:CHAIN:ATOM" -> ("RES", "CHAIN", "ATOM")
    "RES:ATOM" -> ("RES", None, "ATOM")
    "RES" -> ("RES", None, None)
    """
    parts = s.split(':')
    if len(parts) == 3:
        return (parts[0].upper(), parts[1].upper(), parts[2].upper())
    elif len(parts) == 2:
        return (parts[0].upper(), None, parts[1].upper())
    else:
        return (parts[0].upper(), None, None)

# --- atomic masses for a quick-and-dirty COM (add more if needed) -------
_MASS = dict(
    H=1.008,   C=12.011, N=14.007, O=15.999, S=32.06,
    P=30.974,  FE=55.845,  # add Zn, Mg, etc. if your structures need them
)

def _residue_center(residue, mass_weighted: bool = False) -> np.ndarray:
    """
    Return geometric centre or centre-of-mass of a Bio.PDB Residue.
    """
    coords = np.array([a.coord for a in residue if a.element != "H"])  # usually skip H
    if coords.size == 0:
        raise ValueError(f"Residue {residue} has no non-hydrogen atoms")

    if not mass_weighted:
        return coords.mean(axis=0)

    masses = np.array([_MASS.get(a.element.upper(), 0.0) for a in residue if a.element != "H"])
    if np.any(masses == 0):
        raise ValueError("Missing masses for some elements in _MASS dict")
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()

def locate_reference(
        struct,
        resname: str,
        atom_name: str | None = None,
        chain_id: str | None = None,
        mass_weighted: bool = False,
):
    """
    Return either an Atom *or* a 3-vector of floats to serve as the centre.

    If `atom_name` is given, returns the Atom.
    If `atom_name is None`, returns the centre (geometric or COM) of the first
    matching residue.
    """
    residues = (
        r for r in struct.get_residues()
        if r.resname == resname and (chain_id is None or r.get_parent().id == chain_id)
    )
    try:
        residue = next(residues)
    except StopIteration:
        raise ValueError(f"No residue '{resname}' found"
                         f"{'' if chain_id is None else f' in chain {chain_id}'}")

    if atom_name:                       # ← explicit atom requested
        try:
            return next(a for a in residue if a.id.strip() == atom_name)
        except StopIteration:
            raise ValueError(f"Residue '{resname}' has no atom '{atom_name}'")

    # --- no atom requested → return centre coordinate
    return _residue_center(residue, mass_weighted=mass_weighted)

def haem_center(struct):
    heme_atoms = [a for a in struct.get_atoms()
                  if a.parent.resname.startswith(("HEM","HEA","HEME"))]
    return np.mean([a.coord for a in heme_atoms], axis=0)

def run_meeko(polymer_pdb: Path, flex_csv: str|None, out_prefix: Path):
    if flex_csv is not None:
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(polymer_pdb),
            "-o", str(out_prefix),
            "-p",  # write _rigid/_flex
            "--flexres", flex_csv,
            "--allow_bad_res",
        ]
        run(cmd)
        return out_prefix.with_name(out_prefix.name+"_rigid.pdbqt"), out_prefix.with_name(out_prefix.name+"_flex.pdbqt")
    else:
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(polymer_pdb),
            "-o", str(out_prefix)+"_rigid",
            "-p"  # write _rigid/_flex
        ]    
        run(cmd)
        return out_prefix.with_name(out_prefix.name+"_rigid.pdbqt")

def merge_rigid(rigid: Path, cofactor_qt: Path, out_qt: Path):
    lines = rigid.read_text().splitlines(True)
    co = [ln for ln in cofactor_qt.read_text().splitlines(True) if ln.startswith(("ATOM", "HETATM"))]
    out_qt.write_text("".join(lines + co + ["END\n"]))

# ---------------------------------------------------------------------------
# CLUSTER UTILS
# ---------------------------------------------------------------------------

def condensed_rmsd(coords: list[np.ndarray]):
    def rmsd(a, b):
        return np.sqrt(((a - b) ** 2).sum(axis=1).mean())
    d = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d.append(rmsd(coords[i], coords[j]))
    return np.array(d)

def cluster_labels(coords: list[np.ndarray], cut: float = 2.0):
    if len(coords) < 2:
        return np.ones(len(coords), int)
    Z = linkage(condensed_rmsd(coords), "single")
    return fcluster(Z, t=cut, criterion="distance")

def _ensure_hetatm(path: Path):
    txt = path.read_text()
    if txt.startswith("ATOM"):
        txt = txt.replace("ATOM  ", "HETATM", 1)  # first line is enough
        path.write_text(txt)

# ---------------------------------------------------------------------------
# ARGPARSE
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Flexible‑receptor Vina docking")
    p.add_argument("--ligand", required=True, help="Ligand PDB input")
    p.add_argument("--protein", required=True, help="Protein PDB input")
    flex = p.add_mutually_exclusive_group(required=False)
    flex.add_argument("--flex_radius", type=float, default=5, help="Radius (Å) around centre to make residues flexible")
    flex.add_argument("--flex_list", help="CSV chain:resseq list to make flexible")
    p.add_argument("--rigid_cofactor", action="append", metavar="RES:CHAIN:RESSEQ", help="Residue kept rigid (repeatable)")
    p.add_argument("--freeze_radius", type=float, default=3, help="Radius (Å) around centre to make residues frozen")
    p.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    p.add_argument("--cpus", "--cpu", type=int, default=1, dest="cpus")
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--poses_per_run", type=int, default=5)
    p.add_argument("--size", nargs=3, type=float, metavar=("X","Y","Z"), default=[20,20,20], help="Box size Å")
    p.add_argument("--center", nargs=3, type=float, metavar=("X","Y","Z"), help="Box centre coordinates")
    p.add_argument("--box_center_ref", metavar="RES[:CHAIN[:ATOM]]", help="Residue (and optional chain/atom) that defines the centre of the box")
    p.add_argument("--blindt", action="store_true", help="60 Å cube around protein centre")
    p.add_argument("--flex_center", metavar="RES[:CHAIN[:ATOM]]", help="Residue (and optional chain/atom) that defines the centre of the flexible shell")
    p.add_argument("--flex_center_ref", metavar="REF_PDB", help="PDB file from which to read the flex-centre residue (default = current protein)")
    p.add_argument("--freeze_center", metavar="RES[:CHAIN[:ATOM]]", help="Residue/atom used as centre for --freeze_radius")

    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Cofactors that must stay rigid (resname, chain, resseq)
# To add another: RIGID_COFATORS.append(("DXC", "B", "301"))  # example
# ---------------------------------------------------------------------
# RIGID_COFATORS = [
#     ("HEM", "A", "999"),           # haeme / heme iron
# ]

def _triplet(s): return tuple(part.upper() for part in s.split(':'))

def main():
    args = get_args()

    _triplet = lambda s: tuple(part.upper() for part in s.split(':'))
    RIGID_COFATORS: list[tuple[str, str, str]] = [
    _triplet(x) for x in (args.rigid_cofactor or [])
    ]

    out = Path("out"); out.mkdir(exist_ok=True)

    lig_pdb = Path(args.ligand); prot_pdb = Path(args.protein)

    prot_struct = load_structure(prot_pdb)
    if args.box_center_ref:
        center_ref_resname, center_ref_chain, center_ref_atom = parse_center(args.box_center_ref)
        centre_ref = locate_reference(prot_struct, center_ref_resname, center_ref_atom, center_ref_chain, mass_weighted=True)    
        centre  = centre_ref.coord if isinstance(centre_ref, Atom.Atom) else centre_ref
        print("BOX CENTER:", centre)
    elif args.center:
        centre = np.array(args.center)
    else:
        centre = centre_of_pdb(prot_pdb)

    # translate ligand
    lig_trans = out/"ligand_tr.pdb"; translate_pdb(lig_pdb, centre, lig_trans)
    lig_qt = out/"ligand.pdbqt"; pdb_to_pdbqt(lig_trans, lig_qt)

    # flexible and frozen residue selection
    struct = PDBParser(QUIET=True).get_structure("R", str(prot_pdb))

    if args.flex_list or args.flex_center:
        if args.flex_list:
            tags = {t.strip() for t in args.flex_list.split(',') if t.strip()}
            flex_set = {r for r in struct.get_residues() if f"{r.get_parent().id}:{r.id[1]}" in tags}
        elif args.flex_center:
            flex_resname, flex_chain, flex_atom = parse_center(args.flex_center)
            if args.flex_center_ref:
                struct_flex_center_ref = PDBParser(QUIET=True).get_structure("R", str(Path(args.flex_center_ref)))
            else:
                struct_flex_center_ref = struct
            flex_ref   = locate_reference(struct_flex_center_ref, flex_resname, flex_atom, flex_chain, mass_weighted=True)
            flex_coord   = flex_ref.coord if isinstance(flex_ref, Atom.Atom) else flex_ref
            flex_set = find_flexible_residues(struct, flex_coord, args.flex_radius)
            if args.freeze_center:
                freeze_resname, freeze_chain, freeze_atom = parse_center(args.freeze_center)
                freeze_ref = locate_reference(struct, freeze_resname, freeze_atom, freeze_chain, mass_weighted=True)
                freeze_coord = freeze_ref.coord if isinstance(freeze_ref, Atom.Atom) else freeze_ref
                flex_set = find_frozen_residues(struct, freeze_coord, args.freeze_radius, True, flex_set)
        flex_csv = ",".join(sorted(f"{r.get_parent().id}:{r.id[1]}" for r in flex_set))
    else:
        flex_csv = None
    if len(flex_set) == 0: # Otherwise we get an error if there are no residues within the specified flex radii
        flex_csv = None

# ---------------------------------------------------------------------
# Build a protein-only PDB for Meeko and a separate file for cofactors
# ---------------------------------------------------------------------
    protein_noCof_raw = out / "protein_noCof_raw.pdb"
    cofactors_pdb     = out / "cofactors.pdb"

    # start from the full protein
    shutil.copyfile(prot_pdb, protein_noCof_raw)

    if RIGID_COFATORS:
        # open handle to collect the stripped residues
        with cofactors_pdb.open("w") as cof_fh: 
            for resname, chain, resseq in RIGID_COFATORS:
                # extract the residue and append to cofactors.pdb
                buf = io.StringIO()
                extract_residue(Path(prot_pdb), buf, resname, chain, resseq)
                cof_fh.write(buf.getvalue())

                # now strip the residue out of protein_noCof_raw
                tmp = out / "tmp.pdb"
                strip_residue(Path(protein_noCof_raw), tmp, resname, chain, resseq)
                tmp.replace(protein_noCof_raw)  

    protein_noCof_fixed = out / "protein_noCof_fixed.pdb"
    fix_pdb_missing_atoms(protein_noCof_raw, protein_noCof_fixed)
    
    # run_meeko call
    receptor_prefix = out / "receptor"
    out_paths = run_meeko(protein_noCof_fixed, flex_csv, receptor_prefix)
    rigid_qt, flex_qt = (out_paths if isinstance(out_paths, tuple) else (out_paths, None))
    print(rigid_qt)
    # After run_meeko() returns: rigid_qt, flex_qt already exist -if flex is true-
    rigid_full_qt = out / "receptor_full_rigid.pdbqt"
    if RIGID_COFATORS:
        cofactors_qt = out / "cofactors.pdbqt"
        _ensure_hetatm(cofactors_pdb)
        pdb_to_pdbqt(cofactors_pdb, cofactors_qt, assign_charges=False)
        merge_rigid(rigid_qt, cofactors_qt, rigid_full_qt)
    else:
        pdb_to_pdbqt(rigid_qt, rigid_full_qt, assign_charges=False)

    # override box if blindt
    box_size = [60,60,60] if args.blindt else args.size

    # docking loop
    all_coords, all_energies, pose_map = [], [], []
    for run_i in range(args.n_runs):
        v = Vina(sf_name="vina", cpu=args.cpus, seed=run_i)
        # v.set_receptor(str(rigid_full_qt), str(flex_qt))
        if flex_qt is None:
            v.set_receptor(str(rigid_full_qt))
        else:
            v.set_receptor(str(rigid_full_qt), str(flex_qt))
        v.set_ligand_from_file(str(lig_qt))
        v.compute_vina_maps(center=centre.tolist(), box_size=box_size)
        v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.poses_per_run)

        poses_qt = out/f"poses_{run_i}.pdbqt"; v.write_poses(str(poses_qt), n_poses=args.poses_per_run)
        coords = v.poses(n_poses=args.poses_per_run, coordinates_only=True)
        energies = [e[0] for e in v.energies(n_poses=args.poses_per_run)]
        offset = len(all_coords)
        all_coords.extend(coords); all_energies.extend(energies)
        pose_map.extend((run_i, idx) for idx in range(args.poses_per_run))

    labels = cluster_labels(all_coords)
    cluster_inds = collections.defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_inds[lab].append(idx)

    if not cluster_inds:
        logging.warning("No clusters found; exiting.")
        sys.exit(0)

    for cl, idxs in cluster_inds.items():
        best = min(idxs, key=lambda i: all_energies[i])
        run_i, pose_idx = pose_map[best]
        src_qt = out/f"poses_{run_i}.pdbqt"
        # keep ligand + flex residues when you harvest the representative block
        block = src_qt.read_text().split("ENDMDL")[pose_idx] + "ENDMDL\n"
        rep_qt  = out / f"cluster{cl}_rep.pdbqt"
        rep_qt.write_text(block)

        # convert to PDB without losing flex residues
        ob_convert(rep_qt, out / f"cluster{cl}_rep.pdb",
                   "pdbqt", "pdb", ["-xr"])

    # Summarize clusters
    summary = sorted(
        ((lab, len(idxs), min(all_energies[i] for i in idxs))
         for lab, idxs in cluster_inds.items()),
        key=lambda x: x[2]
    )
    with open(out / 'cluster_summary.txt', 'w') as fh:
        fh.write(f"{'cluster':>7s} {'pop':>5s} {'best_E':>10s}\n")
        for lab, pop, e in summary:
            fh.write(f"{lab:7d} {pop:5d} {e:10.3f}\n")

    logging.info("Docking complete: results in ./out")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        logging.error("Subprocess failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.exception(exc)
        sys.exit(1)


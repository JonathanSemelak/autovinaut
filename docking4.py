#!/usr/bin/env python3
"""
Flexible‑receptor docking & clustering pipeline (v0.2)
=====================================================

Features
--------
* Ligand translation to protein centre (or haem centre with `--box_center_heme`).
* Flexible‑receptor preparation via **Meeko**; non‑standard cofactors (e.g. heme) are kept rigid and merged back.
* Multi‑seed AutoDock Vina runs, clustering by RMSD, representative pose output.
* Command‑line flags compatible with earlier *docking4.py* wrappers.

Requires
~~~~~~~~
RDKit, Open Babel, Meeko, PDBFixer, SciPy, NumPy, BioPython, Vina ≥ 1.2.
"""
from __future__ import annotations

import argparse
import collections
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from openbabel import openbabel
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


def pdb_to_pdbqt(pdb: Path, pdbqt: Path):
    ob_convert(pdb, pdbqt, "pdb", "pdbqt", ["--partialcharge", "gasteiger"])


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
# ---------------------------------------------------------------------------
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_pdb_missing_atoms(src: Path, dst: Path):
    """Use PDBFixer to add missing heavy atoms/hydrogens before Meeko."""
    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues(); fixer.findMissingAtoms()
    fixer.addMissingAtoms(); fixer.addMissingHydrogens(pH=7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(dst, "w"), keepIds=True)


# ---------------------------------------------------------------------------

def find_flexible_residues(struct, centre: np.ndarray, radius: float):
    """Return Residue objects whose any atom lies within *radius* Å of *centre* and is a standard AA."""
    search = NeighborSearch(list(struct.get_atoms()))
    near_atoms = search.search(list(centre), radius)
    return {at.get_parent() for at in near_atoms if at.get_parent().resname.upper() in STD_AA}


# ---------------------------------------------------------------------------

def strip_residue(src: Path, dst: Path, resname: str, chain: str, resseq: str):
    with src.open() as fi, dst.open("w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] == resname and ln[21:22] == chain and ln[22:26].strip() == resseq:
                continue
            fo.write(ln)


def extract_residue(src: Path, dst: Path, resname: str, chain: str, resseq: str):
    with src.open() as fi, dst.open("w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] == resname and ln[21:22] == chain and ln[22:26].strip() == resseq:
                fo.write(ln)
        fo.write("END\n")


def run_meeko(polymer_pdb: Path, flex_csv: str, out_prefix: Path):
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

# ---------------------------------------------------------------------------
# ARGPARSE
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Flexible‑receptor Vina docking")
    p.add_argument("--ligand", required=True, help="Ligand PDB input")
    p.add_argument("--protein", required=True, help="Protein PDB input")

    flex = p.add_mutually_exclusive_group(required=True)
    flex.add_argument("--flex_radius", type=float, help="Radius (Å) around centre to make residues flexible")
    flex.add_argument("--flex_list", help="CSV chain:resseq list to make flexible")

    p.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    p.add_argument("--cpus", "--cpu", type=int, default=1, dest="cpus")
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--poses_per_run", type=int, default=5)

    p.add_argument("--size", nargs=3, type=float, metavar=("X","Y","Z"), default=[20,20,20], help="Box size Å")
    p.add_argument("--center", nargs=3, type=float, metavar=("X","Y","Z"), help="Box centre coordinates")
    p.add_argument("--box_center_heme", action="store_true", help="Centre box on haem co‑factor")
    p.add_argument("--blindt", action="store_true", help="60 Å cube around protein centre")

    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    out = Path("out"); out.mkdir(exist_ok=True)

    lig_pdb = Path(args.ligand); prot_pdb = Path(args.protein)

    # determine docking centre
    if args.box_center_heme:
        # take first haem atom as centre
        with prot_pdb.open() as f:
            coords = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                               for ln in f if ln.startswith(("ATOM","HETATM")) and ln[17:20]=="HEM"])
        centre = coords.mean(axis=0)
    elif args.center:
        centre = np.array(args.center)
    else:
        centre = centre_of_pdb(prot_pdb)

    # translate ligand
    lig_trans = out/"ligand_tr.pdb"; translate_pdb(lig_pdb, centre, lig_trans)
    lig_qt = out/"ligand.pdbqt"; pdb_to_pdbqt(lig_trans, lig_qt)

    # flexible residue selection
    struct = PDBParser(QUIET=True).get_structure("R", str(prot_pdb))
    if args.flex_list:
        tags = {t.strip() for t in args.flex_list.split(',') if t.strip()}
        flex_set = {r for r in struct.get_residues() if f"{r.get_parent().id}:{r.id[1]}" in tags}
    else:
        flex_set = find_flexible_residues(struct, centre, args.flex_radius)

    # strip haem, prepare polymer
    polymer_raw = out/"protein_noheme_raw.pdb"; strip_residue(prot_pdb, polymer_raw, "HEM", "A", "999")
    polymer_pdb = out/"protein_noheme.pdb"; fix_pdb_missing_atoms(polymer_raw, polymer_pdb)

    flex_csv = ",".join(sorted(f"{r.get_parent().id}:{r.id[1]}" for r in flex_set))
    rigid_qt, flex_qt = run_meeko(polymer_pdb, flex_csv, out/"receptor")

    # haem as rigid cofactor
    haem_pdb = out/"heme.pdb"; extract_residue(prot_pdb, haem_pdb, "HEM", "A", "999")
    haem_qt = out/"heme.pdbqt"; ob_convert(haem_pdb, haem_qt, "pdb", "pdbqt", ["--partialcharge","gasteiger","-xr"])

    full_rigid_qt = out/"receptor_full_rigid.pdbqt"; merge_rigid(rigid_qt, haem_qt, full_rigid_qt)

    # override box if blindt
    box_size = [60,60,60] if args.blindt else args.size

    # docking loop
    all_coords, all_ener, pose_map = [], [], []
    for run_i in range(args.n_runs):
        v = Vina(sf_name="vina", cpu=args.cpus, seed=run_i)
        v.set_receptor(str(full_rigid_qt), str(flex_qt))
        v.set_ligand_from_file(str(lig_qt))
        v.compute_vina_maps(center=centre.tolist(), box_size=box_size)
        v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.poses_per_run)

        poses_qt = out/f"poses_{run_i}.pdbqt"; v.write_poses(str(poses_qt), n_poses=args.poses_per_run)
        coords = v.poses(n_poses=args.poses_per_run, coordinates_only=True)
        energies = [e[0] for e in v.energies(n_poses=args.poses_per_run)]
        offset = len(all_coords)
        all_coords.extend(coords); all_ener.extend(energies)
        pose_map.extend((run_i, idx) for idx in range(args.poses_per_run))

    labels = cluster_labels(all_coords)
    clusters: dict[int,list[int]] = collections.defaultdict(list)
    for i, lb in enumerate(labels): clusters[lb].append(i)

    for cl, idxs in clusters.items():
        best = min(idxs, key=lambda i: all_ener[i])
        run_i, pose_idx = pose_map[best]
        src_qt = out/f"poses_{run_i}.pdbqt"
        block = src_qt.read_text().split("ENDMDL")[pose_idx].strip()+"\nENDMDL\n"
        rep_qt = out/f"cluster{cl}_rep.pdbqt"; rep_qt.write_text(block)
        ob_convert(rep_qt, out/f"cluster{cl}_rep.pdb", "pdbqt", "pdb")

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


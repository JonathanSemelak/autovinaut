# AutoVinaut

_A lightweight command-line wrapper around **AutoDock Vina 1.2** and **Meeko** for quick, scriptable docking jobs._

* **Rigid and flexible receptor docking**
* Retains rigid **cofactors** (e.g. heme, FAD) with a single flag
* Automatic or explicit **grid box** placement
* Shell-based or explicit-list **flexible residues**
* Multi-run / multi-pose execution + optional clustering

All outputs go to an **`out/`** folder (poses, cluster summary, logs).

---

## 1 · Installation

```bash
# create a fresh environment (example)
conda create -n docking python=3.10
conda activate docking

# core packages
conda install -c conda-forge rdkit openbabel biopython pdbfixer openmm vina scipy

# Meeko (flexible-receptor support)
pip install git+https://github.com/forlilab/Meeko.git
```

> Tested with Python 3.10, AutoDock Vina 1.2.5, RDKit 2023.03, Open Babel 3.1, Meeko 0.5.

---

## 2 · Command-line synopsis

```text
usage: autovinaut.py --ligand LIGAND.pdb --protein PROTEIN.pdb [options]

Box placement
-------------
--blindt                     60×60×60 Å cube centred on receptor
--center        X Y Z        Explicit box centre (Å)
--box_center_ref RES[:CHAIN[:ATOM]]
                             Centre on residue / atom (e.g. HEM:A:FE)
--size          X Y Z        Box size (Å)  [default 20 20 20]

Docking control
---------------
--exhaustiveness N           Vina exhaustiveness       [default 8]
--n_runs         N           Independent Vina seeds    [default 1]
--poses_per_run  N           Poses per run             [default 5]
--cpus           N           CPU threads               [default 4]

Cofactors & flexibility
-----------------------
--rigid_cofactor  RES:CHAIN:RESSEQ     (repeatable)
--flex_radius     R        Shell radius (Å)
--flex_center     RES[:CHAIN[:ATOM]]   Origin of flex shell
--flex_center_ref file.pdb             Coordinates for flex origin
--flex_list       CSV      Explicit list (A:45,B:102,…)
--freeze_center   RES[:CHAIN[:ATOM]]   Origin of rigid shell
--freeze_radius   R        Radius to _exclude_ from flex (Å)
```

---

## 3 · Usage examples

### 3.1 Rigid-only (blind docking)

```bash
python autovinaut.py     --ligand lig.pdb     --protein prot.pdb     --blindt     --exhaustiveness 8 --cpus 8
```

### 3.2 Rigid protein **+ heme cofactor**

```bash
python autovinaut.py     --ligand lig.pdb     --protein prot.pdb     --rigid_cofactor HEM:A:999     --box_center_ref HEM:A:FE     --size 22 22 22     --n_runs 3 --poses_per_run 10
```

### 3.3 Rigid heme **+ flexible shell** around an external ligand pose

```bash
python autovinaut.py     --ligand lig.pdb     --protein prot.pdb     --rigid_cofactor HEM:A:999     --box_center_ref HEM:A:FE     --flex_radius 6     --flex_center LIG     --flex_center_ref pose.pdb     --freeze_center HEM:A:FE     --freeze_radius 2.5     --exhaustiveness 12 --n_runs 5
```

---

## 4 · Output overview

| Path / file                               | Description                                   |
|-------------------------------------------|-----------------------------------------------|
| `out/ligand.pdbqt`                        | Converted ligand                              |
| `out/receptor_full_rigid.pdbqt`           | Receptor + rigid cofactors                    |
| `out/poses_runX_*.pdbqt`                  | Raw Vina poses                                |
| `out/clusterY_rep.pdb` / `.pdbqt`         | Representative pose per RMSD cluster          |
| `out/cluster_summary.txt`                 | Cluster ID, size, best energy                 |
| `out/*.log`                               | Timestamped execution logs                    |

---

## 5 · Tips

* Large boxes → increase `--exhaustiveness` or use more `--cpus`.
* Keep the flexible shell tight to avoid slow runs.
* Repeat `--rigid_cofactor` for multiple prosthetic groups.

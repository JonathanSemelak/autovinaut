#!/bin/bash

LIGAND=testosterone
PROTEIN=F87A
LIGAND_POSE=cluster_5_rep

mkdir -p outputs
cd outputs

python ../docking5.py \
    --ligand           ../inputs/${LIGAND}.pdb \
    --protein          ../inputs/${PROTEIN}.pdb \
    --flex_center      UNL \
    --flex_center_ref  ../inputs/${LIGAND_POSE}.pdb \
    --flex_radius      2 \
    --box_center_ref   HEM:A:FE \
    --freeze_center    HEM:A:FE \
    --freeze_radius    2.5 \
    --rigid_cofactor   HEM:A:999 \
    --exhaustiveness   8 \
    --cpus             4 \
    --n_runs           2 \
    --poses_per_run    5



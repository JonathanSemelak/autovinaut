#!/bin/bash

LIGAND=testosterone
PROTEIN=F87A

mkdir -p outputs
cd outputs

python ../docking4.py \
        --ligand ../inputs/${LIGAND}.pdb \
        --protein ../inputs/${PROTEIN}.pdb \
        --flex_radius 6 \
        --exhaustiveness 8 \
        --cpus 4 \
        --n_runs 2 \
        --poses_per_run 5 \
        --box_center_heme          # <‑‑ new


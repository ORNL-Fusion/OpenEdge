#!/usr/bin/env python3
"""
Analyze PMI impact CSV logs from the CPC benchmark.

Reads all per-rank CSV files (output/pmi_impacts.csv.rank*) and computes:
  - Gross deposition (total absorptions across all ranks)
  - Reflection / sputter / absorb counts and fractions
  - Impact energy and angle statistics
  - Comparison to analytic gross deposition

Usage:
    python3 analyze_pmi.py [csv_base]

    csv_base: base path for CSV files (default: output/pmi_impacts.csv)
              Reads csv_base.rank0, csv_base.rank1, etc.
"""

import sys
import glob
import numpy as np

csv_base = sys.argv[1] if len(sys.argv) > 1 else "output/pmi_impacts.csv"

# Read all rank files
files = sorted(glob.glob(f"{csv_base}.rank*"))
if not files:
    print(f"No files found matching {csv_base}.rank*")
    sys.exit(1)

print(f"Reading {len(files)} rank file(s)...")

# Columns: timestep,rank,surf_id,species,E_in_eV,theta_deg,outcome,E_out_eV,
#           x,y,z,vx,vy,vz,nx,ny,nz
all_rows = []
for f in files:
    with open(f) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            all_rows.append(parts)

n_total = len(all_rows)
if n_total == 0:
    print("No impact events found.")
    sys.exit(1)

# Parse outcomes
outcomes = [r[6] for r in all_rows]
E_in = np.array([float(r[4]) for r in all_rows])
theta = np.array([float(r[5]) for r in all_rows])
E_out = np.array([float(r[7]) for r in all_rows])
species = [r[3] for r in all_rows]

n_R = outcomes.count("R")
n_S = outcomes.count("S")
n_A = outcomes.count("A")

print(f"\n{'='*50}")
print(f"PMI Impact Summary")
print(f"{'='*50}")
print(f"Total impacts:    {n_total}")
print(f"  Reflect (R):    {n_R}  ({100*n_R/n_total:.1f}%)")
print(f"  Sputter (S):    {n_S}  ({100*n_S/n_total:.1f}%)")
print(f"  Absorb  (A):    {n_A}  ({100*n_A/n_total:.1f}%)")
print()
print(f"Gross deposition (absorptions): {n_A}")

# Analytic comparison
# grossDep = nP * (1-R) / (1 - R - Y)
# From the impacts: R_eff = n_R / n_total, Y_eff = n_S / n_total
R_eff = n_R / n_total if n_total > 0 else 0
Y_eff = n_S / n_total if n_total > 0 else 0
A_eff = n_A / n_total if n_total > 0 else 0
print(f"\nEffective coefficients from simulation:")
print(f"  R_eff = {R_eff:.4f}")
print(f"  Y_eff = {Y_eff:.4f}")
print(f"  A_eff = {A_eff:.4f}  (1-R-Y = {1-R_eff-Y_eff:.4f})")

# Analytic with nominal values
nP = 10000
R_nom, Y_nom = 0.5, 0.1
gross_analytic = nP * (1 - R_nom) / (1 - R_nom - Y_nom)
print(f"\nAnalytic gross deposition (R={R_nom}, Y={Y_nom}, nP={nP}):")
print(f"  grossDep = nP*(1-R)/(1-R-Y) = {gross_analytic:.0f}")
print(f"  Simulated:  {n_A}")
if gross_analytic > 0:
    print(f"  Error:      {100*(n_A - gross_analytic)/gross_analytic:+.1f}%")

# Energy statistics
print(f"\n{'='*50}")
print(f"Impact Energy Statistics (all impacts)")
print(f"{'='*50}")
print(f"  E_in  mean={np.mean(E_in):.1f} eV, median={np.median(E_in):.1f} eV, "
      f"max={np.max(E_in):.1f} eV")
print(f"  theta mean={np.mean(theta):.1f} deg, median={np.median(theta):.1f} deg")

# Reflected energy
mask_R = np.array([o == "R" for o in outcomes])
if mask_R.any():
    print(f"\nReflected particles:")
    print(f"  E_in  mean={np.mean(E_in[mask_R]):.1f} eV")
    print(f"  E_out mean={np.mean(E_out[mask_R]):.1f} eV")
    print(f"  RE_eff = <E_out>/<E_in> = {np.mean(E_out[mask_R])/np.mean(E_in[mask_R]):.3f}")

# Species breakdown
unique_sp = sorted(set(species))
if len(unique_sp) > 1:
    print(f"\n{'='*50}")
    print(f"Species Breakdown")
    print(f"{'='*50}")
    for sp in unique_sp:
        mask = [s == sp for s in species]
        n_sp = sum(mask)
        n_sp_A = sum(1 for i, m in enumerate(mask) if m and outcomes[i] == "A")
        print(f"  {sp:6s}: {n_sp} impacts, {n_sp_A} absorbed")

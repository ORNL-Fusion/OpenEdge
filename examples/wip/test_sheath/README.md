# test_sheath

**Data download required:** `./download_data.sh test_west_axi` (from repo root; uses plasma.h5 from test_west_axi)

Standalone analytical sheath validation (Borodkina-style) before OpenEdge integration.

## Script
- `test_borodkina_sheath.py`

## What it uses
- Analytical plasma inputs:
  - `ne` [m^-3]
  - `Te` [eV]
  - `Ti` [eV]
  - `B` [T]
  - angle `alpha`
  - ion mass
  - parallel ion transport as either:
    - `u_par` [m/s], or
    - `Gamma_par` [m^-2 s^-1] with `Gamma = n_i * u_par`

## Quick run (analytical defaults)
```bash
cd examples/test_sheath
python3 test_borodkina_sheath.py --out output/borodkina_sheath_test.png
```

## Example with explicit WEST-like analytical values
```bash
python3 test_borodkina_sheath.py \
  --te 30 --ti 30 --ne 1e19 --b 3.2 \
  --alpha 88 --angle-def normal --mi-amu 2 \
  --gamma-par 3e19 \
  --out output/borodkina_west_like.png
```

The script prints:
- sheath scales (`lambda_D`, `rho_i`, `L_mps`, `c_s`)
- sheath-entrance estimates from potential thresholds (`-Te`, `-3Te`)
- Chodura/Bohm checks based on `u_par` and projected normal speed.

By default it now reads `Te`, `Ti`, `ne`, and `upar=parr_flow` from:
- `../test_west_axi/input/plasma.h5`
and selects a sheath-entrance cell as the point closest to `|u_par/c_s - 1|`,
with:
- `c_s = sqrt((Te + Ti) * e / (2 m_D))`.

Normalization in the plots is done with ion Larmor radius (`d/rho_i`).

## Alpha sweep plot (potential + ne)
Single figure with two panels (left: `phi`, right: `ne`) for multiple `alpha` values:
```bash
python3 test_borodkina_sheath.py \
  --paper-case \
  --alpha-sweep 0,45,85 \
  --alpha-sweep-out output/borodkina_alpha_0_45_85.png
```

To use only manual scalar inputs (no plasma file):
```bash
python3 test_borodkina_sheath.py --use-plasma-h5 no --te 30 --ti 30 --ne 1e20
```

To also plot sheath entrance on WEST `R-Z` grid:
```bash
python3 test_borodkina_sheath.py \
  --alpha-sweep 0,45,85 \
  --plot-rz yes \
  --mach-band 0.8,1.2 \
  --near-layers 3 \
  --rz-out output/sheath_entrance_rz.png
```

You can tighten/loosen the sheath-entry candidate region with:
- `--mach-band min,max` (e.g. `0.9,1.1`)
- `--near-layers N` (wall-adjacent grid layers).

For higher resolution near the wall (`d=0`), add:
```bash
--near-power 3 --npts 3000
```

## EIRENE-like wall-local sheath (WEST 2D)
This computes sheath quantities on each wall segment using WEST
`plasma.h5 + bfield.h5 + wall.txt`:
- EIRENE-like sheath energy drop `E_sheath` (multi-ion)
- Chodura geometry diagnostics (`M_parallel`, `M_normal`, `alpha`, `b·n`)

```bash
python3 test_borodkina_sheath.py \
  --use-plasma-h5 yes \
  --eirene-wall yes \
  --plasma-h5 ../test_west_axi/input/plasma.h5 \
  --bfield-h5 ../test_west_axi/input/bfield.h5 \
  --wall-file ../test_west_axi/input/wall.txt \
  --eirene-out output/eirene_sheath_west.png \
  --eirene-csv output/eirene_sheath_west.csv
```

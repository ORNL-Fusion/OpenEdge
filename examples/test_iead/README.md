# test_iead — Ion Energy-Angle Distribution validation

Validates the OpenEdge sheath model against an independent Fortran Boris
tracker for Ta²⁺/Ta³⁺/Ta⁴⁺ ions in a magnetised Chodura sheath.

## Physics

- Te = Ti = 5 eV, ni = 1×10¹⁹ m⁻³, B = 0.5 T
- Species: Ta²⁺ (8%), Ta³⁺ (62%), Ta⁴⁺ (30%), mass = 180.948 amu
- Background D⁺ sets sheath scales: λ_D ≈ 5.26 µm, ρ_D ≈ 0.644 mm
- Floating potential: φ_float ≈ 18.7 eV (Stangeby formula)
- Tilt angles: α = 0°, 45°, 85° from wall normal

## Files

| File | Description |
|------|-------------|
| `create_case.py` | Generates wall surface, particles, and input files |
| `compare_iead.py` | Compares OpenEdge vs Fortran IEADs (plots + summary) |
| `run_all.sh` | Runs all three angles sequentially |
| `plasma.species` | Species definitions (Ta²⁺, Ta³⁺, Ta⁴⁺ with charges) |
| `fortran/sheath_tracker.f90` | Reference Fortran+OpenMP Boris tracker |
| `fortran/sheath_tracker.py` | Python version of the tracker (slower, for prototyping) |

## Quick start

### 1. Generate Fortran reference data

```bash
cd fortran
gfortran -O3 -fopenmp -o sheath_tracker sheath_tracker.f90
OMP_NUM_THREADS=8 ./sheath_tracker
mv iead_alpha*.csv ..
cd ..
```

### 2. Generate and run OpenEdge cases

```bash
python3 create_case.py
./run_all.sh
```

### 3. Compare

```bash
python3 compare_iead.py
```

Produces `output/iead_comparison.png`, `output/iead_1d_energy.png`,
`output/iead_1d_angle.png`, and `output/iead_summary.txt`.

## Sheath modes

The input files use `kick yes` by default (sheath energy applied as velocity
boost at wall).  To test the spatially-resolved Coulette-Manfredi model,
edit `create_case.py` and change the sheath line to:

```
model coulette_manfredi mD_amu {mD_amu} pot_mult 0
```

Both modes produce <1% energy error vs Fortran at all angles.  See
`src/OPENEDGE/README.md` for full sheath documentation.

## Expected results

| α | OpenEdge <E> | Fortran <E> | OpenEdge <θ> | Fortran <θ> |
|---|-------------|-------------|-------------|-------------|
| 0° | 61.0 eV | 61.0 eV | 14.6° | 14.8° |
| 45° | 61.5 eV | 61.4 eV | 17.7° | 18.3° |
| 85° | 61.9 eV | 61.4 eV | 26.6° | 36.1° |

The 85° angle difference is physical: OpenEdge tracks full 3D gyro-orbits
while the Fortran code uses a 1D sheath potential along z.

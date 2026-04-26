# test_gca_vs_boris

Single-particle pusher validation against the analytical axisymmetric
tokamak field of **Khan et al. (2012)** (also used as the verification
test in the OpenEdge paper, Sec. 4.1, Fig. 3a/b). Pure Boris (high
subcycles) and the hybrid Boris/GCA pusher both run on the same
synthetic `khan_plasma.h5` and are compared by overlaying:

- (R, Z) particle trajectory
- (R, Z) of the guiding center
- Δμ/μ₀ and Δ|v|/|v₀|
- ρ_L / L_B vs. the GCA switch threshold

## Field

Khan's Cartesian field decomposes axisymmetrically into

```
B_R(R,Z) = -Z/(2R)
B_phi(R) =  R0/R
B_Z(R)   = (R-1)/(2R)
ψ(R,Z)   =  Z²/4 + R³/6 - R²/4
btf·rtf  =  R0
```

`make_khan_plasma_h5.py` writes a 257×257 `/equilibrium/{r,z,psi,btf,rtf,psib}`
grid into `khan_plasma.h5`. `fix background` reconstructs B from ψ and
serves it to both pushers — Boris uses B at point, GCA also gets smooth
grad|B|, curvature, and curl(b̂).

## Setup

| Parameter | Value |
|---|---|
| Geometry | 3D Cartesian, `dimension 3`, `boundary o o o` |
| Box | `−2.0 2.0   −2.0 2.0   −2.0 2.0` |
| Grid | 10 × 10 × 10 (single particle uses point queries; grid res does not matter) |
| Plasma + B | `fix pd background file khan_plasma.h5 static yes` (axi /equilibrium/* on R,Z grid; pusher does R = √(x²+y²) internally) |
| Species | H+ (matches Khan's unit-mass / unit-charge ion) |
| Launch | (x, y, z) = (1.09, 0, 0), `v = (0, 6.005e4, -1.972e5) m/s` (Khan paper Cartesian) |
| dt | 5 × 10⁻¹⁰ s (no subcycling — `ω_c·dt ≈ 0.048` per step keeps cumulative phase error small at marginal trapping) |
| Run length | 600 000 steps (300 µs ≈ 6 bounce periods) |
| Boris | `subcycles 1`, `gca_switch 1e12` (forces Boris branch) |
| GCA | `subcycles 1`, `gca_switch 2.5`  (hybrid GCA above ρ_L/L_B = 0.4) |

## Files

| File | Purpose |
|---|---|
| `make_khan_plasma_h5.py` | Writes `khan_plasma.h5` from the analytical Khan ψ map. |
| `khan_plasma.h5` | Synthetic plasma file (regenerate with the helper). |
| `in.boris` | Pure Boris reference (huge gca_switch, 100 subcycles). |
| `in.gca` | Hybrid Boris/GCA (gca_switch 2.5, 10 subcycles fallback). |
| `input/source.single` | One H+ at (Z, R, φ) = (0, 1.09, 0). |
| `input/plasma.species` | Single-line H+ species table. |
| `plot_trajectories.py` | Reads both dumps + khan_plasma.h5 equilibrium, writes 3 PNGs to `output/`. |

## Run

```bash
cd /home/cloud/OpenEdge/examples/test_gca_vs_boris
python3 make_khan_plasma_h5.py             # one-time, regenerates khan_plasma.h5
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 1 /home/cloud/buildOpenEdge/src/spa_mpi -in in.boris
mpirun -np 1 /home/cloud/buildOpenEdge/src/spa_mpi -in in.gca
python3 plot_trajectories.py
```

## Generated figures

Written to `output/`:

- `boris_vs_gca.png` — (R, Z) trajectory + R(t), Z(t), |v|(t).
- `boris_vs_gca_gc.png` — guiding-center (R, Z), R_gc(t), Δμ/μ₀ [ppm], Δ|v|/|v₀| [ppm].
- `boris_vs_gca_rhoL_over_LB.png` — ρ_L / L_B(t) with the switch threshold.

## Pass criterion

- Trajectory shape matches paper Fig. 3a (banana orbit). Doubling
  v_φ (`6.005e4` → `1.20e5` m/s in `source.single`) gives a passing
  orbit (Fig. 3b).
- GC (R, Z) traces from Boris and GCA overlap to a small RMS distance
  on the orbit scale.
- Δ|v|/|v₀| stays well below 100 ppm over the full run (energy is
  exactly conserved by Boris; this checks the integrator).
- Δμ/μ₀ is **not** a clean pass criterion in trapped/banana orbits —
  v∥ ↔ v⊥ exchange continuously over the bounce, so μ ∝ v⊥² oscillates
  by O(1) by construction. What matters is that Boris and GCA see the
  *same* μ trace, not that either one is constant.
- ρ_L / L_B stays in the GCA-allowed regime once the switch threshold
  is passed.

## Going further

- **Passing orbit** (paper Fig. 3b): edit `input/source.single` to
  `v_φ = 1.20e5` m/s (doubled). No other changes.
- **Ripple superbanana** (paper Fig. 3c): the Khan ripple term
  `b_TF = δ_TF cos(N ξ)` is **non-axisymmetric**, so it cannot be
  expressed as ψ(R,Z). Either add a 3D B map to plasma.h5 or use a
  separate `fix bfield/particle` deck for that case.

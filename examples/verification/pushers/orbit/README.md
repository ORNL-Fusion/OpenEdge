# verification/pushers/orbit

Single-particle pusher verification against the analytical axisymmetric
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
| GCA | `subcycles 1`, `gca_switch 2.5` (hybrid GCA above ρ_L/L_B = 0.4), selectable RK2/RK4 integrator |

## GCA integrators

The full Littlejohn GCA path keeps both midpoint RK2 and RK4:

```text
global pusher ... gca_integrator rk2
global pusher ... gca_integrator rk4
```

- `rk2` evaluates the full RHS twice using the explicit midpoint method.
- `rk4` evaluates the same full RHS four times and remains the C++ default for
  backward compatibility.
- `simple` remains available as a reduced one-stage update, but omits the
  full B-star curvature/curl(b) terms.

`rk2` and `rk4` resample E, B, and their derivatives at each RK stage
position (`Pusher::sample_gca_fields`); a failed stage query falls back to
the frozen k1 fields. The old frozen-field formulation carried a secular
energy drift (+9100 ppm Δ|v|/|v₀| over this 600k-step orbit, rk2 and rk4
byte-identical); with stage resampling the drift is bounded (|Δv/v| max
~140 ppm over the bounce, final ~1 ppm) and the two integrators differ as
they should. `simple` still freezes fields over its single stage.

## Files

| File | Purpose |
|---|---|
| `make_khan_plasma_h5.py` | Writes `khan_plasma.h5` from the analytical Khan ψ map. |
| `khan_plasma.h5` | Synthetic plasma file (regenerate with the helper). |
| `in.boris` | Pure Boris reference (subcycles 1, ω_c·dt ≈ 0.048; huge gca_switch). |
| `in.gca` | 3D hybrid Boris/GCA with command-line-selectable RK2 or RK4. |
| `in.gca.2d` | Same orbit, pure GCA, 2D Cartesian slots (x=R, y=Z). |
| `in.gca.axi` | Same orbit, pure GCA, 2D axisymmetric slots (x=Z, y=R). |
| `input/source.single` | One H+ for the 3D decks. |
| `input/source.2dcart`, `input/source.axi` | Same launch in the 2D slot layouts. |
| `input/plasma.species` | Single-line H+ species table. |
| `plot_trajectories.py` | Compares any GCA dump against `traj.boris`; PASS/FAIL gates. |

## Run

Single-particle verification: run with `-np 1` for reproducible gate
numbers (`./check_mpi.sh` verifies 1-vs-4-rank agreement to dump
precision; see `../README.md`).

```bash
cd examples/verification/pushers/orbit
python3 input/make_khan_plasma_h5.py       # one-time, regenerates khan_plasma.h5
SPA=~/build_oe/src/spa_mac_mpi   # or your build
mpirun -np 1 $SPA -in in.boris
mpirun -np 1 $SPA \
  -var gcaIntegrator rk2 -in in.gca
mpirun -np 1 $SPA \
  -var gcaIntegrator rk4 -in in.gca
python3 plot_trajectories.py --gca-dump traj.gca.rk2 --tag rk2
python3 plot_trajectories.py --gca-dump traj.gca.rk4 --tag rk4
# 2D variants (same orbit, same Boris reference):
mpirun -np 1 ... -var gcaIntegrator rk4 -in in.gca.2d
mpirun -np 1 ... -var gcaIntegrator rk4 -in in.gca.axi
python3 plot_trajectories.py --gca-dump traj.gca2d.rk4  --tag 2d.rk4  --mode 2dcart
python3 plot_trajectories.py --gca-dump traj.gcaaxi.rk4 --tag axi.rk4 --mode axi
```

## Generated figures

Written to `output/`:

- `traj.gca.rk2` and `traj.gca.rk4` — raw particle trajectories for the two
  full-GCA integrators.
- `boris_vs_gca_<tag>.png` — (R, Z) trajectory + R(t), Z(t), |v|(t).
- `boris_vs_gca_gc_<tag>.png` — guiding-center (R, Z), R_gc(t), Δμ/μ₀ [ppm], Δ|v|/|v₀| [ppm].
- `boris_vs_gca_rhoL_over_LB_<tag>.png` — ρ_L / L_B(t) with the switch threshold.

## Pass criterion

- Trajectory shape matches paper Fig. 3a (banana orbit). Doubling
  v_φ (`6.005e4` → `1.20e5` m/s in `source.single`) gives a passing
  orbit (Fig. 3b).
- GC (R, Z) traces from Boris and GCA: RMS < 1e-2 m. (Boris reference
  discretization bound: dt vs dt/2 GC RMS = 1.7e-3 m.)
- Energy gate on the stored GC invariant H = ½m·v∥² + μB(X_gc) from the
  dumped `p_gca_*` state: fitted secular |ΔH/H| < 20 ppm and bounded
  excursion max |ΔH/H| < 500 ppm. All six configs (3d/2dcart/axi ×
  rk2/rk4) measure sub-ppm secular drift.
- The *stored* GCA μ is invariant by construction — the energy gate above
  is the meaningful integrator check. μ *reconstructed* from x,v dumps
  oscillates O(1) over a banana bounce (v∥ ↔ v⊥ exchange) for Boris and
  GCA alike; only same-trace agreement matters there.
- ρ_L / L_B stays in the GCA-allowed regime once the switch threshold
  is passed.

## Going further

- **Passing orbit** (paper Fig. 3b): edit `input/source.single` to
  `v_φ = 1.20e5` m/s (doubled). No other changes.
- **Ripple superbanana** (paper Fig. 3c): the Khan ripple term
  `b_TF = δ_TF cos(N ξ)` is **non-axisymmetric**, so it cannot be
  expressed as ψ(R,Z). Either add a 3D B map to plasma.h5 or use a
  separate `fix bfield/particle` deck for that case.

# fix grain/ablate (droplet/evaporate)

Grain heating, phase change and evaporation/sublimation; spawns the vapor as
kinetic atoms.

    fix ID grain/ablate Nevery MIXTURE background PD [keywords]

Keywords:

- `material NAME` — grain material from the registry / `material` command
  (default `Li`). Supplies rho, cp, atom mass, latent heat, Antoine
  vapor-pressure coefficients (`log10 p[atm] = a + b/T`), emissivity,
  T_melt, h_melt.
- `heating flux|oml` — heat source. `flux` (default): 0.25·|q| from the
  plasma.h5 `q_par`/`q_perp` fields. `oml`: electron/ion OML collection
  from local ne/Te/Ti with sheath transmission (zeta = 2.5) and surface
  recombination; uses the OML charge from `grain/charge` when present.
  Use `oml` for grains flying through the SOL volume.
- `ion_mass_amu M` — background ion mass for the OML ion flux (default 2.0).
- `twall_K T` — wall temperature for radiative cooling (default 300).
- `heatflux/scale S` — multiplier on the heat source (default 1.0).
- `rocket_eta E` — ablation-recoil asymmetry in [0,1] (default 0, off);
  kick along −grad(Te).
- `emit_into MIX` — spawn evaporated atoms as particles of this mixture.

Model: Antoine + Hertz–Knudsen flux; dR/dt = −m_atom·Γ/ρ;
dT/dt = 3(Q − εσ(T⁴−T_wall⁴) − Γ·ΔH/N_A)/(ρ·Cp_eff·R) with an
apparent-heat-capacity melting band at T_melt. Adaptive substepping
(≤25 K, ≤2% radius per substep). Grains die at 10% of the species-file
mass. If `grain/emit nweight` is used, the vapor source and the
cumulative-atom tally (`compute_scalar`, MPI-reduced) scale by the
per-grain weight.

2D/axi only. See `grain_material.h` for the material registry.

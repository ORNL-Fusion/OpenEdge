# Choosing `global fnum` for OpenEdge neutral-transport cases

`fnum` is the statistical weight — each simulated particle represents `fnum`
real particles. It's the knob that trades statistics for speed.

## Recipe

Pick `fnum` to target a steady-state **simulated** particle count in the
desired range (typically 10k–100k). From first principles, at steady state:

```
N_sim_steady = (total_emission_rate / fnum) * mean_lifetime_sim_sec
            ≈ launch_rate_sim_per_step * mean_lifetime_steps
```

so

```
fnum ≈ total_emission_rate [real/s] * mean_lifetime [s]  /  N_sim_target
```

The three inputs you need:

1. **`total_emission_rate`** — all sources summed in real particles/second.
   For a recycling case: the plasma ion flux × wall area integrated, in
   particles/s. Typically within a factor of 2 of the total ionization rate
   in the volume (flux in ≈ flux absorbed in steady state).
   For DIII-D SOLPS-EIRENE reference: **~5 × 10²⁷ /s**.

2. **`mean_lifetime`** — time from emission to removal, whichever of these
   happens first:
     * ionization: τ_iz ≈ 1/(n_e ⟨σv⟩_iz). For T_e ≈ 20 eV, n_e = 10²⁰ m⁻³:
       τ_iz ≈ 1/(10²⁰ × 10⁻¹⁴) = **10⁻⁶ s**.
     * transit (puff exits through boundary/wall): L / v_n ≈ 0.5 m / 10⁴ m/s
       = 5 × 10⁻⁵ s.
     * wall return path: typically much longer.
   Use the shorter of these: for hot-core neutrals τ_iz dominates;
   for cold-divertor puffs transit dominates.
   For DIII-D: a decent average is **~10⁻⁵ s** (mix of cold targets and
   hot SOL).

3. **`N_sim_target`** — sim-particle count you want alive at steady state.
   50 000 is a good sweet spot: enough for smooth source maps, cheap
   enough to run on a few dozen MPI ranks.

Putting the numbers together:

```
fnum ≈ 5e27 × 1e-5 / 5e4 ≈ 1e18
```

## Sanity check at runtime

```
N_sim at step N = steady-state value → fnum chosen correctly
N_sim growing unbounded → fnum too low (ionization slower than emission)
N_sim ≈ 0 → fnum too high (almost nothing emitted)
```

The DIII-D case in this directory uses `fnum = 1e13` for a *smaller*
divertor-puff test (not full device flux); scale up to `1e17–1e18` once
you're emitting from all divertor lines at SOLPS-matched rates.

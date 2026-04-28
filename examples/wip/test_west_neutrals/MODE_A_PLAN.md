# Mode A (EIRENE-semantics) implementation plan

Target: make `fix chem/adas` behave like EIRENE in a SOLPS-coupled call.
Launches a fixed number of neutrals from sources (puff + recycling),
tracks each to termination (ionization / wall absorb), returns
cell-averaged source terms. Frozen plasma per call (already what
`fix plasma/data static yes` gives us).

Current `neutral` branch provides: source tallies (count, momentum,
energy per reaction) + per-species moments. What's missing is **neutral
termination + stop-when-exhausted**, which is the heart of Mode A.

---

## Design summary

**Key realization**: currently `fix chem/adas` handles ionization by
relabeling the particle (`ip->ispecies = rchosen->products[0]`).
For Mode A we just need to ALSO kill the particle -- no new allocation,
no new data structures. SPARTA has `Particle::compress_reactions()` for
this exact use case.

No Boris / sheath changes needed: those only act on charged particles
(`charge != 0`). With D+ removed on ionization, the only remaining
tracked species are neutrals, and their code paths are already free-flight.

---

## Code changes (by file)

### 1. `src/OPENEDGE/fix_chem_adas.h`

Add protected members:

```cpp
int eirene_mode;                       // 0 = kinetic D+, 1 = terminate on ionize
int stop_on_exhaust;                   // 0 = run full requested steps, 1 = stop when source species exhausted
std::vector<int> source_species;       // species indices tagged as "injected neutrals" (for exhaustion check)
std::vector<int> dellist;              // per-step particle-removal indices
char **src_species_names;              // names parsed from kw args
int nsrc_species;
```

### 2. `src/OPENEDGE/fix_chem_adas.cpp`

Constructor: parse new keyword arguments
```
fix ID chem/adas nevery Z reactions_file ...
    mode neutral                                  <-- new
    source_species D D2                           <-- new
    stop_on_exhaust yes                           <-- new
```

In `attempt()`, replace the ionization species swap with conditional kill:
```cpp
if (rchosen->type == IONIZATION && eirene_mode) {
  // Tally source (already happens above)
  // Mark particle for deletion -- do NOT change ispecies.
  dellist.push_back(particle_index);
  return 1;
}
// existing code path (species relabel) for CX, dissociation, and Mode B
ip->ispecies = rchosen->products[0];
```

The particle-index needs to be threaded through to `attempt()`. Either
add a parameter, or expose a member variable, or move the deletion logic
back up into `end_of_step_no_average()`.

At the end of `end_of_step_no_average()`:
```cpp
if (!dellist.empty()) {
  particle->compress_reactions(dellist.size(), dellist.data());
  dellist.clear();
}
```

### 3. Stop-when-exhausted

After compression, count alive source-species particles:
```cpp
bigint alive_local = 0;
for (int i = 0; i < particle->nlocal; i++) {
  int sp = particle->particles[i].ispecies;
  for (int src : source_species) {
    if (sp == src) { alive_local++; break; }
  }
}
bigint alive_global = 0;
MPI_Allreduce(&alive_local, &alive_global, 1, MPI_SPARTA_BIGINT,
              MPI_SUM, world);
if (alive_global == 0 && stop_on_exhaust) {
  // SPARTA checks this each step
  update->runflag = 0;
}
```

Source-species indices are resolved in `init()` from user-supplied names:
```cpp
for (int i = 0; i < nsrc_species; i++) {
  int sp = particle->find_species(src_species_names[i]);
  if (sp < 0) error->all(FLERR, "fix chem/adas source_species not found");
  source_species.push_back(sp);
}
```

### 4. Reset-tally (`fix_modify`)

Add `modify_param()` override so the user can reset cumulative tallies
between coupling iterations without re-reading the input:
```
fix_modify fchem reset_tally
```

In `modify_param()`:
```cpp
if (strcmp(arg[0], "reset_tally") == 0) {
  if (array_grid && maxgrid_src > 0)
    memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * 20);
  return 1;
}
```

### 5. (Optional) Volumetric recombination source

In Mode A, D+ isn't a kinetic particle. Pure EIRENE computes recombination
as a NEUTRAL source drawn from the fluid ion density:

```
dn_D / dt |_rec = n_e * n_i^fluid * <sigma v>_rec
```

Would need a new emission mechanism: `fix emit/volume` that spawns D
particles in each cell at the local recombination rate. Not needed for
initial demo -- recombination is 6 orders of magnitude below ionization
at SOL temperatures.

### 6. (Optional) Plasma-driven wall recycling source

Pure EIRENE takes a D+ wall-flux map from SOLPS and emits D at Franck-
Condon energies from each wall element. Would need `fix emit/surf` to
consume an external flux field (currently it reads a constant rate).

For a first Mode A demo, the HYBRID approach works: keep kinetic D+
tracking only for wall recycling (current `surf_react recycle`
infrastructure), and terminate kinetic D+ born from VOLUMETRIC ionization.

---

## Effort estimate

| Step | File | Effort |
|---|---|---|
| 1. Keyword parsing (`mode neutral`, source_species, stop_on_exhaust) | fix_chem_adas.cpp constructor + init() | 2 h |
| 2. Terminate particle on ionization | fix_chem_adas.cpp attempt() + end_of_step | 2 h |
| 3. `Particle::compress_reactions` call + dellist management | fix_chem_adas.cpp | 1 h |
| 4. Stop-when-exhausted check | fix_chem_adas.cpp end_of_step | 2 h |
| 5. `fix_modify reset_tally` | fix_chem_adas.cpp | 1 h |
| 6. Mirror edits in `src/` copies (per CLAUDE.md) | — | trivial |
| 7. Test case: test_west_neutrals with `mode neutral` + `fix emit/surf` puff | examples/ | 2 h |
| 8. Doc updates | `doc/openedge/fix_chem_adas.txt` | 1 h |
| **Subtotal (core Mode A)** | | **~1.5 days** |
| 9. (Optional) Volumetric recombination source | new fix emit/volume | ~1 day |
| 10. (Optional) Plasma-driven wall recycling | extend fix emit/surf | ~1 day |

---

## How to use from an input deck

Target user experience after these changes land:

```
# Plasma (frozen per call, provided by Gkeyll / SOLPS)
fix pd plasma/data file plasma.h5 static yes

# D2 gas puff at specified wall location (e.g. inner divertor)
group surf puffgroup surf <IDs near R=2.189 Z=-0.693>
fix fpuff emit/surf puffgroup D2 rate 4.84e20 temp 0.03

# Wall recycling
surf_collide wallColl diffuse 500.0 1.0
surf_react wallRecycle recycle wall.recycle
surf_modify wall collide wallColl react wallRecycle

# Chemistry in EIRENE mode
fix fchem chem/adas 1 1 reactions.txt adas_dir ../../database/adas &
    mode neutral &
    source_species D D2 &
    stop_on_exhaust yes

# Output: per-cell source tallies (20 cols) and moments
compute cmom grid all species n u v w temp
fix fmom ave/grid all 1 100 1000 c_cmom[*] ave running

dump dgrid grid all 1000 output/grid id xc yc f_fchem[*] f_fmom[*]

# Run: stops automatically when all D/D2 have been ionized or absorbed
run 1000000 pre no post no
```

The `run 1000000` upper bound is a safety cap; the actual run terminates
via `stop_on_exhaust` when the neutral population goes to zero.
Source tallies and moments are then ready for Gkeyll handoff.

---

## Testing plan

1. **Unit test**: tiny 2D box, 1000 D particles, constant plasma.
   - Count ionizations via `tally_reactions[]`.
   - Verify particle count drops to 0 within expected time (tau_iz ~ 10 us).
   - Verify `stop_on_exhaust` triggers termination.
2. **WEST demo**: puff from (2.189, -0.693) as in S3X-EIRENE.
   - Compare ionization source map to S3X-EIRENE output.
   - Compare neutral penetration depth.
3. **Performance**: compare Mode A (no D+ tracked) to Mode B (kinetic D+)
   on the same WEST case. Mode A should be significantly faster since
   Boris subcycling dominates in Mode B.

---

## Open design questions

1. **Per-particle kill vs species-gated kill**: should we add a per-
   species flag so, say, D ionization is terminal but W ionization
   continues kinetically? Would let the same fix handle main species
   + impurities in one run. Probably yes.

2. **Weighted sampling / variance reduction**: EIRENE uses stratified
   sampling with split/roulette. OpenEdge has `fix particle/weight`;
   whether to integrate split/roulette is an open question for later.

3. **Particle census for time-dependent coupling**: if we want the
   time-dependent EIRENE mode (not just SOLPS-style steady-state),
   we need to keep unterminated particles between calls. The
   `fix plasma/data reload_plasma()` call already gives us the
   'new plasma' signal -- could be the trigger to reset-tally +
   continue. Defer.

---

## References in code

- Ionization species relabel: `src/OPENEDGE/fix_chem_adas.cpp:640` (`ip->ispecies = rchosen->products[0]`)
- SPARTA particle-deletion API: `src/particle.cpp:362` (`Particle::compress_reactions()`)
- Source-tally infrastructure: `src/OPENEDGE/fix_chem_adas.cpp:660` (20-column `array_grid`)
- Stop-the-run API: `update->runflag` (grep for usage)
- S3X-EIRENE reference: `/home/cloud/S3XE/WEST_MULTISPEC/eirene_coupling.txt`
- S3X puff location: `/home/cloud/S3XE/WEST_MULTISPEC/mesh/puffs` line 4
  (tri 22252, side 2 -> R = 2.1887, Z = -0.6925)

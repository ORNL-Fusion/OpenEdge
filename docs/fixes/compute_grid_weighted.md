# `compute grid/weighted` — per-particle-weighted grid moments

Identical in scope to the base `compute grid` but uses each particle's
per-particle weight `pweight` (set by `fix particle/weight`) in place
of the global `fnum`. Required for source-driven cases where emission
rates vary across the wall — without the weighted compute, density
columns are wrong by the same factor that varies pweight.

## Syntax

```
compute ID grid/weighted <grid-group> <mix-ID> value1 value2 ...
```

`fix particle/weight` must already be defined; the compute reads the
custom attribute `pweight` at every step.

## Values

Output columns mirror `compute grid`, with a `_w` suffix to make the
weighting explicit. Each column is computed per mixture group.

### Density-like (extensive — normalised by cell volume)

| keyword | formula |
|---|---|
| `n_w` | $\sum p_i$ |
| `nrho_w` | $W \sum p_i / V$ |
| `massrho_w` | $W \sum p_i\, m_i / V$ |
| `pxrho_w`, `pyrho_w`, `pzrho_w` | $W \sum p_i\, m_i\, v_{\alpha,i} / V$ |
| `kerho_w` | $W \sum \tfrac{1}{2} p_i\, m_i\, v_i^2 / V$ |

Here $p_i \equiv \texttt{pweight}_i$, $W$ is the cell weight
`cinfo[icell].weight`, and $V$ is the cell volume.

### Mean / intensive (per-particle moments)

| keyword | formula |
|---|---|
| `mass_w` | $\sum p_i\, m_i \,/\, \sum p_i$ |
| `u_w`, `v_w`, `w_w` | $\sum p_i\, m_i\, v_{\alpha,i} \,/\, \sum p_i\, m_i$ |
| `usq_w`, `vsq_w`, `wsq_w` | $\sum p_i\, m_i\, v_{\alpha,i}^2 \,/\, \sum p_i\, m_i$ |
| `ke_w` | $\tfrac{1}{2}\, \texttt{mvv2e}\, \sum p_i\, m_i\, v_i^2 \,/\, \sum p_i$ |
| `temp_w` | $\dfrac{\texttt{mvv2e}}{3 k_B}\, \sum p_i\, m_i\, v_i^2 \,/\, \sum p_i$ |
| `erot_w`, `evib_w` | $\sum p_i\, \varepsilon_{\mathrm{rot},i} \,/\, \sum p_i$ etc. |
| `trot_w`, `tvib_w` | $\dfrac{2\,\texttt{mvv2e}}{k_B}\, \sum p_i\, \varepsilon_{\cdot,i} \,/\, \sum p_i\, g_i$ |

The `tvib` / `trot` denominator $g_i$ is the per-particle internal-DOF
count (`species[ispecies].rotdof` or `vibdof`), matching the
upstream `compute grid` convention.

### Group fractions (cell-level denominator across all groups)

| keyword | formula |
|---|---|
| `nfrac_w` | $\dfrac{\sum_{\text{group}} p_i}{\sum_{\text{cell}} p_i}$ |
| `massfrac_w` | $\dfrac{\sum_{\text{group}} p_i\, m_i}{\sum_{\text{cell}} p_i\, m_i}$ |

`nfrac_w` and `massfrac_w` allocate cell-level tally columns
(`cellcount_w`, `cellmass_w`) populated over **all** mixture groups,
while every other keyword tallies per-group only.

## Example

A typical Li source diagnostic on a per-surf-emission case:

```
fix pw particle/weight
fix femit_evp surface/emit/source LiSource divertor cevap \
    perspecies no normal yes nlaunch_total 200 model thermal_tsurf Tsurf_lm

mixture LiAll Li Li+ Li2+ Li3+

compute n_grid grid/weighted all LiAll \
    nrho_w u_w v_w temp_w nfrac_w

dump d_grid grid all 100 grid_dump.txt id \
    c_n_grid[*]
```

## Implementation notes

- File: `src/OPENEDGE/compute_grid_weighted.{cpp,h}` (mirrored to `src/`
  per the three-copy rule).
- Carries the same icell-bounds-check guard as `compute grid` against
  the upstream stale-icell heap-UAF after `fix balance` / `fix adapt`.
- Same `post_process_grid` machinery as upstream — works with
  `fix ave/grid` for time-averaged dumps.
- All prefactors (`mvv2e`, `boltz`) come from `update->`, so unit
  changes (`units si` vs `units cgs`) are picked up automatically.

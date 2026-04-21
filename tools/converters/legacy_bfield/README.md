# legacy_bfield — parked scripts for time-dependent B-field workflows

These two scripts generate a **standalone `bfield.h5`** (regular R-Z grid
with `br`, `bt`, `bz` datasets). The main OpenEdge flow no longer uses
`bfield.h5` — the SOLPS / SOLEDGE3X / OEDGE converters embed the
B-field directly in `plasma.h5`, and `compute plasma/fields` reads it
from there. See commit `ebbd97c` ("Drop bfield.h5: single plasma.h5
carries embedded br/bt/bz") for the rationale.

They're parked here because they're useful for **time-dependent
B-field** runs where the B-field needs to change independently of the
plasma state:

| script | purpose |
|---|---|
| `geqdsk2bfield_h5.py` | Convert a G-EQDSK equilibrium file to a `bfield.h5` on a regular R-Z grid. Useful when you have an EFIT reconstruction and no SOLPS run, or to build a reference B-field snapshot. |
| `gen_bfield_sweep.py`  | Produce a time series of `bfield.h5` files by sweeping poloidal-field coil currents or by shifting an equilibrium. Used for strike-point-sweep studies (see `test_west_timedep`). |

## When to revive

When you want time-dependent B-field coupling you will need to:

1. Restore these scripts to `tools/converters/` (or invoke them in
   place with an explicit path).
2. Re-enable the 2nd-positional bfield argument in
   `compute plasma/fields` — today the parser hard-errors on it (see
   `src/OPENEDGE/compute_plasma_fields.cpp` ~ line 82). The stub code
   for `magneticFieldsPath` and the `readMagneticFieldFileData` /
   `broadcastMagneticData` helpers is still present in the C++ side,
   so wiring it back up is mostly removing the rejection error.
3. Add a time-stepped reload hook (analogous to `openedge_reload_plasma`
   but for bfield.h5 only) so the outer loop can swap bfield files
   without regenerating plasma.h5.

For static equilibria (the current production case) this is
unnecessary — the single plasma.h5 file with embedded B is the right
representation.

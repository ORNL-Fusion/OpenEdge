# DUSTT-2005 uniform-plasma benchmark

Grain charging, drag and kinematics against DUSTT (Pigarov, PoP 12,
122508) in a uniform prescribed plasma (`fix pd background constant`,
B_z = 1 T). No geometry or plasma file, so any deviation is the model
implementation.

Sub-cases: `drag` (OML charging and ion friction vs python integration),
`efield` (F = Z_d e E), `free` (centrifugal kinematics), `neut` (neutral
drag), `see` (secondary electron emission), `vac` (fixed-charge vacuum
acceleration), and a 2-rank repeat of `drag`.

## Run

```bash
./run.sh /path/to/spa_mpi     # all sub-cases + scripts/check_dustt.py; exit 0 = PASS
```

`scripts/make_input.py` regenerates `input/`.

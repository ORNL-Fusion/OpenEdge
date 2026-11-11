# fix particle/tag: core-transit split of wall impacts

`fix ID particle/tag <surf-group>` marks a marker (custom int `transit`) when it hits a surface of the
group — the core cut of a wedge deck — and `surf_react surface/pwi ... ehist` then reports wall impacts
of transited markers separately (`transit` column of the E/A rows, joint `J` rows), so the IEAD at a
contact area splits into locally recycled W and W that went through the core. `in.transit`: population A
hits the target directly (34 eV), population B bounces off a specular "core" plate first (116 eV);
`scripts/check_transit.py` requires every B impact and no A impact to be tagged, marginals and joint
histogram consistent, and the cumulative mode (`ehist ... every 0`) to write one block at run end.

    SPA=<binary> NP=4 ./run.sh

Registered in `regression/run_regression.sh` as `pwi_transit_tag`.

# Emission release schedule

`fix surface/emit/source` evaluates the source once per window of `nevery` steps and
releases the particles of the window over its steps. This case checks the source
accounting: the summed weight of the emitted particles against flux x area x time.

    python3 scripts/check.py        # SPA=<binary>; under a minute

Covered: `nlaunch_total`, `nlaunch` and the default (fnum) mode with `nevery` 1 and 7;
1, 2 and 4 ranks; a rebalance every 5 steps; a restart inside a window. The weighted
modes are exact to the 8 printed digits and give the same marker count on any rank
count; the window cut by a restart may differ by one marker per element.

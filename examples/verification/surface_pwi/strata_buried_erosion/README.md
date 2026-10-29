# strata_buried_erosion

A 1e15 m^-2 B layer under a 2e14 m^-2 Wd deposit on the WEST wedge patch, eroded
by the background (`adens_erosion cpmiB 0 B`), no particles. B lies inside the
reaction zone, so the strata stack must give it up: the `s_adens[B]` column and
the stack drain together, W is exposed, Wd is untouched.

Before the Oct 29 2026 fix, `erode_species` only looked at the top stratum (Wd),
removed nothing, and the column went negative while conc_B stayed frozen: an
inexhaustible source whenever a `deposit_as` layer sits on a coating.

    SPA=... PYTHON=/usr/bin/python3 ./run.sh

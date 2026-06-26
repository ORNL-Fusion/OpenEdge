#!/bin/bash
# Run SOLPS post-processing 'b2plot wlld' on a converged case to generate
# target profile files in <run_dir>/b2pl.exe.dir/:
#   fp_tg_o.dat, fp_tg_i.dat        flux profiles, lower outer / inner target
#   fp_tg_ou.dat, fp_tg_iu.dat      flux profiles, upper outer / inner (CDN/DDN)
#   ld_tg_o.dat, ld_tg_i.dat (etc.) load (heat flux) profiles
#   wlld.dat, peak_power, sput_trg.sum, shadow.dat
#
# Run AFTER SOLPS converges (b2fstate must exist in <run_dir>).
#
# Usage: ./run_b2plot_wlld.sh <run_dir>
#   <run_dir>  SOLPS case directory containing b2fstate (e.g. solps/attached__mist)
#              Defaults to the current directory if omitted.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNDIR="${1:-$PWD}"
RUNDIR="$(cd "$RUNDIR" && pwd)"        # absolute
cd "$RUNDIR"

if [[ ! -s b2fstate ]]; then
    echo "ERROR: no b2fstate in $RUNDIR — run SOLPS first" >&2
    exit 1
fi

# SOLPSWORK is the parent holding baserun/ and the case dirs (b2plot reads
# the shared geometry from there).
SOLPSWORK="$(dirname "$RUNDIR")"

# Wipe any previous b2plot output (b2plot doesn't truncate properly when
# re-run; old files get appended to, corrupting the parsable format).
rm -rf b2pl.exe.dir

# Source SOLPS env (csh-style; use a sub-shell so it doesn't pollute caller).
# PAGER=cat disables the `module list` pager that otherwise blocks on `less`
# when run without a TTY (under nohup); stdin from /dev/null as a belt-and-braces.
SOLPS_DIR=/home/cloud/local/solps/solps-iter-3.0.8-devel
tcsh -c "setenv PAGER cat && setenv MANPAGER cat && setenv MODULES_PAGER cat && cd $SOLPS_DIR && source setup.csh gfortran && setenv SOLPSWORK $SOLPSWORK && cd $RUNDIR && echo '1111 wlld' | b2plot" < /dev/null

echo "wrote $RUNDIR/b2pl.exe.dir/{fp_tg_*,ld_tg_*,wlld,peak_power*,sput_trg.sum}"

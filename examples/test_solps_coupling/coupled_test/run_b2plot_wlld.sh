#!/bin/bash
# Run SOLPS post-processing 'b2plot wlld' on the converged case to
# generate target profile files in solps/v0_1/b2pl.exe.dir/:
#   fp_tg_o.dat, fp_tg_i.dat        flux profiles, lower outer / inner target
#   fp_tg_ou.dat, fp_tg_iu.dat      flux profiles, upper outer / inner (CDN/DDN)
#   ld_tg_o.dat, ld_tg_i.dat (etc.) load (heat flux) profiles
#   wlld.dat, peak_power, sput_trg.sum, shadow.dat
#
# Run AFTER SOLPS converges (b2fstate must exist in solps/v0_1/).
#
# Usage: ./run_b2plot_wlld.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/solps/v0_1"

if [[ ! -s b2fstate ]]; then
    echo "ERROR: no b2fstate in $(pwd) — run SOLPS first" >&2
    exit 1
fi

# Wipe any previous b2plot output (b2plot doesn't truncate properly when
# re-run; old files get appended to, corrupting the parsable format).
rm -rf b2pl.exe.dir

# Source SOLPS env (csh-style; use a sub-shell so it doesn't pollute caller).
SOLPS_DIR=/home/cloud/local/solps/solps-iter-3.0.8-devel
tcsh -c "cd $SOLPS_DIR && source setup.csh gfortran && cd $(pwd) && echo '1111 wlld' | b2plot"

echo "wrote $(pwd)/b2pl.exe.dir/{fp_tg_*,ld_tg_*,wlld,peak_power*,sput_trg.sum}"

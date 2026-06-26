#!/bin/tcsh
# Wrapper to run SOLPS for one case (mode), called by the coupling driver.
# Usage: ./run_solps.csh <run_dir> [nprocs]
#   <run_dir>  SOLPS case directory containing b2mn.dat (e.g. solps/attached)
#   [nprocs]   MPI ranks for b2run (default 1)

set RUNDIR = `pwd`
if ($#argv >= 1) set RUNDIR = $1
set nprocs = 1
if ($#argv >= 2) set nprocs = $2

# Absolute path to the case dir
set RUNDIR = `cd $RUNDIR && pwd`

# Disable interactive pagers. Under nohup (no TTY) the SOLPS env setup runs
# `module list`, which pipes to `less` and blocks forever waiting for a keypress.
# Forcing cat keeps env-setup non-interactive so b2run can launch.
setenv PAGER cat
setenv MANPAGER cat
setenv MODULES_PAGER cat

# Source the SOLPS environment
if ($?OPENEDGE_SOLPS_DIR) then
    set SOLPS_DIR = "$OPENEDGE_SOLPS_DIR"
else
    set SOLPS_DIR = /home/cloud/local/solps/solps-iter-3.0.8-devel
endif
set SAVE_DIR = `pwd`
cd $SOLPS_DIR
source setup.csh gfortran < /dev/null
cd $SAVE_DIR

# SOLPSWORK is the parent that holds baserun/ and the case dirs
setenv SOLPSWORK `dirname $RUNDIR`

cd $RUNDIR

# --- pre-run cleanup: remove regenerated SOLPS outputs so each run starts clean.
# Keeps inputs (b2fgmtry, b2fstati, b2fstate, b2mn.dat, b2.*.parameters,
# b2.sources.profile, source2d.*, she2d.*) and, per request, all fort.* files.
echo "== cleaning SOLPS outputs in $RUNDIR (keeping inputs + fort.*) =="
set nonomatch
rm -rf b2mn.exe.dir
rm -f b2fmovie b2fplasma b2time.nc b2tallies.nc b2mn.prt b2ftrace b2ftrack run.log .quit .pause
rm -f *.last10
unset nonomatch

if ($nprocs > 1) then
    setenv OMPI_FC /usr/bin/gfortran
    source $SOLPSTOP/SETUP/mpi
    # b2run's -m handler scans for embedded quote chars in the argv (it sed-strips
    # `^"` and `"$`). Without backslash-escaping the script-context tcsh strips the
    # quotes too aggressively, leading to a `shift: No more words.` error in the
    # wrapper's mpiloop. The escaped form preserves the quotes inside argv.
    b2run -m \"mpirun -np $nprocs\" b2mn
else
    b2run b2mn
endif

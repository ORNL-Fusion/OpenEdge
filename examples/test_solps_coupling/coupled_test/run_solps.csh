#!/bin/tcsh
# Wrapper to run SOLPS from the coupling driver
# Usage: ./run_solps.csh [nprocs]
# Must be called from the coupled_test directory

set nprocs = 1
if ($#argv >= 1) set nprocs = $1

# Source SOLPS environment from the SOLPS tree
set SOLPS_DIR = /home/cloud/local/solps/solps-iter-3.0.8-devel
set SAVE_DIR = `pwd`

cd $SOLPS_DIR
source setup.csh gfortran
cd $SAVE_DIR

# Set SOLPSWORK to point to our solps directory
setenv SOLPSWORK `pwd`/solps

# Now run from the v0_1 case directory
cd solps/v0_1

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

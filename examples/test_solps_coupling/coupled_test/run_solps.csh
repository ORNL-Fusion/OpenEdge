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
    b2run -m "mpirun -np $nprocs" b2mn
else
    b2run b2mn
endif

# Copy output from b2mn.exe.dir back to run directory
# (coupling driver expects b2fstate here, not inside b2mn.exe.dir/)
if (-f b2mn.exe.dir/b2fstate) then
    cp b2mn.exe.dir/b2fstate .
    cp b2mn.exe.dir/b2fstate b2fstati
endif
if (-f b2mn.exe.dir/b2mn.prt) cp b2mn.exe.dir/b2mn.prt .

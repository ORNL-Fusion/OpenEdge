#!/bin/bash
set -e
cd /home/cloud/OpenEdge/examples/test_solps_coupling/coupled_test
time python3 /home/cloud/OpenEdge/tools/converters/convert_solps_plasma.py solps/v0_1 --plasma-out /tmp/test_plasma.h5 --equ-file solps/baserun/g000001.00001_symm.X4.equ --b2fstate solps/v0_1/b2fstati

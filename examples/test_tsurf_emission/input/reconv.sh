#!/bin/bash
cd /home/cloud/OpenEdge/examples/test_tsurf_emission/input
python3 /home/cloud/OpenEdge/tools/converters/convert_solps_plasma.py /home/cloud/li/v0 --b2fstate /home/cloud/li/v0/b2fstate --equ-file /home/cloud/li/baserun/dg.equ --plasma-out plasma.h5 --wall-in divertor_slab_strip.surf --geometry cart
cp /home/cloud/li/v0/ld_tg_o.dat /home/cloud/OpenEdge/examples/test_tsurf_emission/input/ld_tg_o.dat
echo "OK: plasma.h5 + ld_tg_o.dat updated from /home/cloud/li/v0"

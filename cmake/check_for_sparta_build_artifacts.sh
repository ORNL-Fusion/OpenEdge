#!/bin/bash
cd $1

# Exclude OPENEDGE from the check since its files are committed directly to src/
installed_packages=$(make package-status 2>/dev/null | grep "Installed YES" | grep -v "OPENEDGE" > /dev/null 2>&1; echo $?)
installed_style_files=$([ $(ls -lat style_*.h 2> /dev/null | wc -l) -ge 1 ] && true || false; echo $?)

if [ $installed_packages -eq 0 ]; then
    #if [ $2 == "y" ]; then
	#make no-all
    #else
	echo "CMake may not build properly! Please run 'make no-all' from $1."
    #fi
fi

if [ $installed_style_files -eq 0 ]; then
    #if [ $2 == "y" ]; then
	#rm -f style_*.h
    #else
	echo "CMake may not build properly! Please run 'make purge' from $1."
    #fi
fi

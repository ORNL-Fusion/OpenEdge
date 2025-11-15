#!/usr/bin/env python -i

# Script:  logplot.py
# Purpose: use GnuPlot to plot two columns from a SPARTA log file
# Author:  Steve Plimpton (Sandia), sjplimp at sandia.gov
# Syntax:  logplot.py logfile X Y
#          logfile = SPARTA log file
#          X,Y = plot Y versus X where X,Y are stats keywords
#          once plot appears, you are in Python interpreter, type C-D to exit
import numpy as np
from matplotlib import pyplot as plt

import sys, os
import numpy as np
# Always try env var first, then append your known good paths
tool_paths = [
    os.environ.get("SPARTA_PYTHON_TOOLS"),
    "/Users/42d/sparta/tools/pizza",
    "/Users/42d/sparta/tools",
]

for p in tool_paths:
    if p and os.path.isdir(p):
        sys.path.insert(0, p)

from olog import olog
from gnu import gnu


# --- add this helper ---
def ci_get(o, *keys):
    # map lowercased names to originals
    name_map = {name.lower(): name for name in o.names}
    mapped = []
    for k in keys:
        k_low = k.lower()
        if k_low in name_map:
            mapped.append(name_map[k_low])
            continue
        # unique prefix match
        matches = [name for name in o.names if name.lower().startswith(k_low)]
        if len(matches) == 1:
            mapped.append(matches[0])
        else:
            raise Exception(f"unique log vector {k} not found")
    return o.get(*mapped)
# --- end helper ---

#if len(sys.argv) != 5:
#  raise Exception("Syntax: logplot.py logfile X Y")


#step c_1 c_3 np

logfile = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]
#zlabel = sys.argv[4]
#
#lg = olog(logfile)

## use case-insensitive get instead of lg.get
#step, temp, dens = ci_get(lg, xlabel, ylabel, zlabel)
#
#ev2kelvin = 11604.505
#timestep = 1e-8
#time  = step * np.asarray(timestep)
#
#plt.figure(figsize=(10, 5))
#
#plt.subplot(121)
#plt.plot(time, temp/np.asarray(ev2kelvin), 'r', alpha=0.5, label="raw")
#plt.legend()
#plt.xlabel("Time")
#plt.ylabel("Temperature")
#
#plt.subplot(122)
#plt.plot(time, dens, 'r', alpha=0.5, label="raw")
#plt.legend()
#plt.xlabel("Time")
#plt.ylabel("Density")
#
#plt.tight_layout()
#plt.show()


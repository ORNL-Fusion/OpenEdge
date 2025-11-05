import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from math import floor
import h5py

data = h5py.File("ADAS_Rates_3.h5", "r")
for key in data.keys():
    print(key, data[key].shape)

print(data['gridChargeState_Ionization'][:])
data.close

#Atomic_Number (1,)
#IonizationRateCoeff (8, 48, 26)
#RecombinationRateCoeff (8, 48, 26)

#gridChargeState_Ionization (2, 8)
#gridChargeState_Recombination (2, 8)
#gridDensity_Ionization (26,)
#gridDensity_Recombination (26,)
#gridTemperature_Ionization (48,)
#gridTemperature_Recombination (48,)


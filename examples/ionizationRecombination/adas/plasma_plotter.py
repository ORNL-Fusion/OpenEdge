import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File("ADAS_Rates_74.h5") as f:
    print(10**f["gridDensity_Ionization"][:].min(), 10**f["gridDensity_Ionization"][:].max()/1e16)
#    print(f["IonizationRateCoeff"][:].min(), f["IonizationRateCoeff"][:].max())

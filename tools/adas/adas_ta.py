#!/usr/bin/env python3

# File written by from Patrick Tamin
# Modified by Abdou Diaw

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from math import floor
import h5py

################################### Inputs ###################################

AMFilesDir = '/Users/42d/eirene-db/Database/AMdata/Adas_Eirene_2010/adf11'

use_tantalum_from_tungsten = True
Z_ta = 73

reactions = ['acd', 'scd']
year = '89'#'89'#'96'
elements = ['w']#['ar', 'b','kr','xe']#['be', 'c', 'he', 'li', 'n', 'ne', 'o']
element_nuclear_charges = [74]

AtomicZ = element_nuclear_charges[0]
IonRate = RecRate = None
logDens_acd = logTe_acd = iZmin_acd = iZmax_acd = None
logDens_scd = logTe_scd = iZmin_scd = iZmax_scd = None

##############################################################################

# Number of elements / reactions / files to treat
Nelts = len(elements)
Nreac = len(reactions)
Nfiles = Nelts*Nreac


for iReac in range(Nreac):
    for iElt in range(Nelts):
        reaction = reactions[iReac]   # 'acd' (recomb), 'scd' (ionization)
        element = elements[iElt]
        element_nuclear_charge = element_nuclear_charges[iElt]
        AMFile = f"{reaction}{year}_{element}.dat"
        AMFileFull = f"{AMFilesDir}/{reaction}{year}/{AMFile}"
        FitFile = f"{reaction}_{element_nuclear_charge}.h5"  # not used below


        print(f'Processing file {AMFile}...')

        with open(AMFileFull) as f:
            contents = f.readlines()

        HeaderComment = contents[0]
        nums = [int(s) for s in HeaderComment.split() if s.isdigit()]
        izmax, nDens, nTe, iZmin, iZmax = nums[0], nums[1], nums[2], nums[3], nums[4]
        nZ = iZmax - iZmin + 1
        nData = nDens * nTe

        logQ = np.zeros([nDens, nTe, nZ])
        logDens = np.zeros(nDens)
        logTe = np.zeros(nTe)

        iline = 2
        iDens = 0
        iTe = 0
        while iTe < nTe:
            flist = [float(s) for s in contents[iline].split()]
            nflist = len(flist)
            imax = 0

            if iDens < nDens:
                imax = min(nDens - iDens, nflist)
                logDens[iDens:iDens+imax] = flist[0:imax]
                iDens += imax

            imin = imax
            if (iDens == nDens) and (iTe < nTe) and (imin < nflist):
                imax = min(nTe - iTe, nflist - imin)
                # FIXED slice here:
                logTe[iTe:iTe+imax] = flist[imin:imin+imax]
                iTe += imax

            iline += 1

        for iZ in range(nZ):
            iline += 1  # skip header
            Data1D = np.zeros(nData)
            iData = 0
            while iData < nData:
                flist = [float(s) for s in contents[iline].split()]
                nflist = len(flist)
                imax = min(nData - iData, nflist)
                Data1D[iData:iData+imax] = flist[0:imax]
                iData += imax
                iline += 1
            logQ[:, :, iZ] = np.reshape(Data1D, (nDens, nTe), order='F')

        # Map to the correct physical meaning (recommended mapping)
        if reaction == 'scd':  # ionization
            IonRate = logQ
            logDens_scd, logTe_scd = logDens, logTe
            iZmin_scd, iZmax_scd = iZmin, iZmax
        elif reaction == 'acd':  # recombination
            RecRate = logQ
            logDens_acd, logTe_acd = logDens, logTe
            iZmin_acd, iZmax_acd = iZmin, iZmax

        # Now safe to print and write
if IonRate is None or RecRate is None:
    raise RuntimeError("Missing IonRate or RecRate — check input files and reactions list order.")

print(f"IonRate shape {IonRate.shape}")
print(f"RecRate shape {RecRate.shape}")

# ---- Ta-from-W truncation happens *here*, after both scd and acd are loaded ----
if use_tantalum_from_tungsten:
    # Set the atomic number to Ta
    AtomicZ = Z_ta

    # Ionization (scd): charge axis goes from iZmin_scd .. iZmax_scd
    # We want to stop at Z_ta, so keep states up to Z_ta.
    nZ_keep_ion = Z_ta - iZmin_scd + 1

    if nZ_keep_ion > IonRate.shape[2]:
        raise ValueError(
            f"Requested {nZ_keep_ion} charge states for Ta, but IonRate only has {IonRate.shape[2]}"
        )

    IonRate = IonRate[:, :, :nZ_keep_ion]
    iZmax_scd = iZmin_scd + nZ_keep_ion - 1

    # Recombination (acd) should have compatible charge indexing.
    # Use the same number of charge states on the 3rd axis.
    nZ_keep_rec = nZ_keep_ion
    if nZ_keep_rec > RecRate.shape[2]:
        raise ValueError(
            f"Requested {nZ_keep_rec} charge states for Ta, but RecRate only has {RecRate.shape[2]}"
        )

    RecRate = RecRate[:, :, :nZ_keep_rec]
    iZmax_acd = iZmin_acd + nZ_keep_rec - 1

    print(f"Truncated to Ta (Z={Z_ta}):")
    print(f"  IonRate shape {IonRate.shape}, charge range {iZmin_scd}..{iZmax_scd}")
    print(f"  RecRate shape {RecRate.shape}, charge range {iZmin_acd}..{iZmax_acd}")

# Now build charge grids using the (possibly updated) *_scd/*_acd ranges
grid_charge_ion = np.array([
    np.arange(iZmin_scd, iZmax_scd),         # z
    np.arange(iZmin_scd + 1, iZmax_scd + 1)  # z+1
])

grid_charge_rec = np.array([
    np.arange(iZmin_acd + 1, iZmax_acd + 1), # z+1
    np.arange(iZmin_acd, iZmax_acd)          # z
])

output_filename = f"ADAS_Rates_{AtomicZ}.h5"
with h5py.File(output_filename, 'w') as f:
    f.create_dataset('Atomic_Number', data=np.array([AtomicZ]))
    f.create_dataset('IonizationRateCoeff', data=IonRate.T)
    f.create_dataset('RecombinationRateCoeff', data=RecRate.T)
    f.create_dataset('gridDensity_Ionization', data=logDens_scd)
    f.create_dataset('gridTemperature_Ionization', data=logTe_scd)
    f.create_dataset('gridDensity_Recombination', data=logDens_acd)
    f.create_dataset('gridTemperature_Recombination', data=logTe_acd)
    f.create_dataset('gridChargeState_Ionization', data=grid_charge_ion)
    f.create_dataset('gridChargeState_Recombination', data=grid_charge_rec)


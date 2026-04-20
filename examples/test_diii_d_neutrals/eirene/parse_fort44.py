#!/usr/bin/env python3
"""
Parse an EIRENE fort.44 output file and plot the neutral-density maps.

Layout of fort.44:
  line 1  :  npol  nrad  date  version-hash
  line 2  :  natm  nmol  nion  [npho]
  line 3  :  "<natm> atom species names"   (one per line)
  ...     :  (nmol molecule names, nion ion names)
  then a sequence of tally blocks, each:
     *eirene data field <name> with size <N>
     [N numbers, 5 per line, format 1p5e18.7]

We expose the cell-based (size == npol*nrad) tallies as a dict keyed by name,
reshape to (nrad, npol) (radial, poloidal — row-major Fortran), and plot
n_D, n_D2, n_(D+) densities on the matching (R, Z) grid from eirene_truth.h5.
"""
import os, sys, re
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS = os.path.dirname(os.path.abspath(__file__))

def parse_fort44(path):
    with open(path) as fh:
        tokens = fh.read().split()
    it = iter(tokens)
    def take(n, typ=float):
        return [typ(next(it)) for _ in range(n)]

    npol, nrad = int(next(it)), int(next(it))
    _date, _hash = next(it), next(it)  # discard
    natm, nmol, nion, _npho = int(next(it)), int(next(it)), int(next(it)), int(next(it))
    atom_names = [next(it) for _ in range(natm)]
    mol_names = [next(it) for _ in range(nmol)]
    ion_names = [next(it) for _ in range(nion)]

    # Now a stream of "*eirene data field <name> with size <N>" + N numbers.
    # Easiest approach: re-read the file as lines, split by the '*eirene' marker.
    with open(path) as fh:
        text = fh.read()
    hdr = re.compile(r'^\*eirene data field\s+(\S.*?)\s+with size\s+(\d+)\s*$', re.MULTILINE)
    tallies = {}
    matches = list(hdr.finditer(text))
    for k, m in enumerate(matches):
        name = m.group(1).strip()
        n = int(m.group(2))
        start = m.end()
        stop = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        data_text = text[start:stop]
        # data_text may contain species-name lines ("D", "D2", "D+") between
        # header and numbers. Extract only numeric tokens.
        vals = np.array(re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?',
                                   data_text), dtype=float)
        if vals.size < n:
            raise ValueError(f'tally {name!r}: expected {n}, got {vals.size}')
        tallies[name] = vals[:n]
    return {
        'npol': npol, 'nrad': nrad,
        'atom_names': atom_names,
        'mol_names':  mol_names,
        'ion_names':  ion_names,
        'tallies': tallies,
    }

def reshape_cell(v, nrad, npol):
    """fort.44 cell tallies are stored column-major in EIRENE (pol-inner)."""
    return v.reshape((nrad, npol), order='F')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS, 'fort.44')
    d = parse_fort44(path)
    nrad, npol = d['nrad'], d['npol']
    print(f'fort.44: {nrad} radial x {npol} poloidal  '
          f'({nrad*npol} cells)')
    print(f'species: atoms={d["atom_names"]}, '
          f'molecules={d["mol_names"]}, ions={d["ion_names"]}')

    # Pull the cell-based neutral-density tallies
    cells = {k: v for k, v in d['tallies'].items() if v.size == nrad * npol}
    print(f'cell-based tallies ({len(cells)}): {list(cells.keys())[:10]}...')

    # Load (R,Z) from eirene_truth.h5 (extracted from balance.nc).
    truth = os.path.join(THIS, '..', 'input', 'eirene_truth.h5')
    with h5py.File(truth, 'r') as f:
        R_all = f['R'][:]
        Z_all = f['Z'][:]
        S_iz_truth = f['S_iz'][:]
    # truth is shape (ny, nx) = (nrad+2, npol+2) with EIRENE "additional" ghost
    # cells on each side. Strip one ring so it becomes (nrad, npol).
    assert R_all.shape == (nrad + 2, npol + 2), (R_all.shape, nrad, npol)
    R = R_all[1:-1, 1:-1]
    Z = Z_all[1:-1, 1:-1]

    # Key fields
    n_D  = reshape_cell(cells['dab2'],  nrad, npol)
    n_D2 = reshape_cell(cells['dmb2'],  nrad, npol)
    n_Dp = reshape_cell(cells['dib2'],  nrad, npol)
    T_D  = reshape_cell(cells['tab2'],  nrad, npol)
    srcml = reshape_cell(cells['srcml'], nrad, npol)

    print(f'n_D   peak = {n_D.max():.3e} m^-3')
    print(f'n_D2  peak = {n_D2.max():.3e} m^-3')
    print(f'n_D+  peak = {n_Dp.max():.3e} m^-3 (test-ion, should be tiny)')
    print(f'T_D   range = {T_D.min():.2f}..{T_D.max():.2f} eV')
    print(f'srcml peak = {srcml.max():.3e}  (molecular source)')

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)

    def panel(ax, data, title, vmin=None, cmap='viridis'):
        pos = data[data > 0]
        if pos.size and vmin is None:
            vmin = max(pos.min(), 1e10)
        if pos.size == 0:
            im = ax.pcolormesh(R, Z, data, cmap=cmap, shading='nearest')
        else:
            im = ax.pcolormesh(R, Z, np.maximum(data, vmin),
                               norm=LogNorm(vmin=vmin, vmax=pos.max()),
                               cmap=cmap, shading='nearest')
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.12)
        ax.set_aspect('equal'); ax.set_xlabel('R [m]'); ax.set_title(title)

    panel(axes[0], n_D,   r'$n_D$ [m$^{-3}$]', vmin=1e14)
    panel(axes[1], n_D2,  r'$n_{D_2}$ [m$^{-3}$]', vmin=1e14)
    panel(axes[2], srcml, 'srcml [m$^{-3}$s$^{-1}$] (mol. source)', vmin=1e14)
    panel(axes[3], S_iz_truth[1:-1, 1:-1],
          r'$S_{iz}$ EIRENE truth [m$^{-3}$s$^{-1}$]', vmin=1e18)
    axes[0].set_ylabel('Z [m]')
    plt.suptitle('Standalone EIRENE (DIII-D, fort.44) — neutral density + sources',
                 y=1.01)
    plt.tight_layout()
    out = os.path.join(THIS, '..', 'output', 'eirene_standalone.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'wrote {out}')

if __name__ == '__main__':
    main()

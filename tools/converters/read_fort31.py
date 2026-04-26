"""
Python port of SOLPS-routines/Matlab/reading/read_ft31.m (J.D. Lore).

fort.31 is written by SOLPS-ITER's coupling driver as the per-B2-cell
plasma input for EIRENE. It carries the same `(nx+2, ny+2)` grid as
`b2fstate`, plus additional derived quantities:

    dnib     ion density                                     [m^-3]
    uub      poloidal velocity                               [m/s]
    vvb      radial velocity                                 [m/s]
    wwb      toroidal velocity                               [m/s]
    teb      electron temperature    (stored in J, convert)  [eV]
    tib      ion temperature         (stored in J, convert)  [eV]
    prb      pressure                                        [Pa]
    upb      parallel velocity                               [m/s]
    rrb      pitch angle b_x / b_tot                         [-]
    fnixb    poloidal ion flow, left face                    [#/s]
    fniyb    radial  ion flow, bottom face                   [#/s]
    feixb    poloidal ion heat flux, left face               [W]
    feiyb    radial  ion heat flux, bottom face              [W]
    feexb    poloidal electron heat flux, left face          [W]
    feeyb    radial  electron heat flux, bottom face         [W]
    uudiab   diamagnetic-drift ion velocity, poloidal        []
    vvdiab   diamagnetic-drift ion velocity, radial          []
    pob      electric potential                              [V]
    volb     cell volume                                     [m^3]
    bfeldb   |B|                                             [T]
    bpolb    B_poloidal                                      [T]
    bradb    B_radial                                        [T]
    btorb    B_toroidal                                      [T]
    zib      average ion charge (Z_eff per cell)             [-]

All arrays are returned in NumPy with Fortran-order shape `(nx+2, ny+2)`
for the scalar blocks and `(nx+2, ny+2, ns_plasma)` for the per-species
blocks. `ns_plasma` excludes neutrals (species with `zamax == 0`), same
as SOLPS-ITER's own fort.31 writer.

Usage:
    from read_fort31 import read_fort31
    ft31 = read_fort31("/path/to/run_dir/fort.31", nx=96, ny=36,
                        ns_plasma=1)
    print(ft31.bfeldb.shape)                  # (98, 38)
    print(ft31.teb[5, 10])                    # Te in eV at cell (5,10)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

QE_J_PER_EV = 1.602176634e-19


@dataclass
class Fort31Data:
    # Per-species (ns_plasma)
    dnib: np.ndarray
    uub: np.ndarray
    vvb: np.ndarray
    wwb: np.ndarray
    upb: np.ndarray
    fnixb: np.ndarray
    fniyb: np.ndarray
    uudiab: np.ndarray
    vvdiab: np.ndarray
    # Scalar
    teb: np.ndarray
    tib: np.ndarray
    prb: np.ndarray
    rrb: np.ndarray
    feixb: np.ndarray
    feiyb: np.ndarray
    feexb: np.ndarray
    feeyb: np.ndarray
    pob: Optional[np.ndarray] = None
    volb: Optional[np.ndarray] = None
    bfeldb: Optional[np.ndarray] = None
    bpolb: Optional[np.ndarray] = None
    bradb: Optional[np.ndarray] = None
    btorb: Optional[np.ndarray] = None
    delta_sheathxb: Optional[np.ndarray] = None
    delta_sheathyb: Optional[np.ndarray] = None
    zib: Optional[np.ndarray] = None
    # Bookkeeping
    nx: int = 0
    ny: int = 0
    ns_plasma: int = 0
    file: str = ""
    truncated: bool = field(default=False)


def _pull(buf: np.ndarray, ptr: int, shape: tuple) -> tuple[np.ndarray, int]:
    """Read `prod(shape)` floats from the flat buffer into an F-order array."""
    n = int(np.prod(shape))
    if ptr + n > len(buf):
        raise EOFError(f"fort.31 truncated: need {n} more values, have {len(buf) - ptr}")
    arr = buf[ptr:ptr + n].reshape(shape, order="F")
    return arr, ptr + n


def _pull_optional(buf: np.ndarray, ptr: int, shape: tuple):
    """Same as _pull, but returns (None, ptr) if not enough data left."""
    n = int(np.prod(shape))
    if ptr + n > len(buf):
        return None, ptr
    arr = buf[ptr:ptr + n].reshape(shape, order="F")
    return arr, ptr + n


def read_fort31(path: str | Path, nx: int, ny: int,
                ns_plasma: int) -> Fort31Data:
    """Read fort.31, returning all blocks in SI units (Te/Ti in eV).

    Parameters
    ----------
    path : str | Path
        Path to fort.31 (or a directory containing it).
    nx, ny : int
        B2 real-cell counts (not including ghost).
    ns_plasma : int
        Number of charged (plasma) species, EXCLUDING neutrals. SOLPS-
        ITER's convention: count species with `zamax > 0` only.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "fort.31"
    if not p.exists():
        raise FileNotFoundError(f"fort.31 not found: {p}")

    # Slurp the whole file as floats (robust to weird line wrapping).
    vals: list[float] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    buf = np.array(vals, dtype=np.float64)

    # Block shapes
    shape_s = (nx + 2, ny + 2)                      # scalar
    shape_n = (nx + 2, ny + 2, ns_plasma)           # per plasma species

    ptr = 0
    dnib, ptr   = _pull(buf, ptr, shape_n)
    uub,  ptr   = _pull(buf, ptr, shape_n)
    vvb,  ptr   = _pull(buf, ptr, shape_n)
    wwb,  ptr   = _pull(buf, ptr, shape_n)
    teb_J, ptr  = _pull(buf, ptr, shape_s)
    tib_J, ptr  = _pull(buf, ptr, shape_s)
    prb,  ptr   = _pull(buf, ptr, shape_s)
    upb,  ptr   = _pull(buf, ptr, shape_n)
    rrb,  ptr   = _pull(buf, ptr, shape_s)
    fnixb, ptr  = _pull(buf, ptr, shape_n)
    fniyb, ptr  = _pull(buf, ptr, shape_n)
    feixb, ptr  = _pull(buf, ptr, shape_s)
    feiyb, ptr  = _pull(buf, ptr, shape_s)
    feexb, ptr  = _pull(buf, ptr, shape_s)
    feeyb, ptr  = _pull(buf, ptr, shape_s)
    uudiab, ptr = _pull(buf, ptr, shape_n)
    vvdiab, ptr = _pull(buf, ptr, shape_n)

    # Convert energy blocks from J to eV (matches Jeremy's read_ft31).
    teb = teb_J / QE_J_PER_EV
    tib = tib_J / QE_J_PER_EV

    # Tail blocks are optional (some SOLPS-ITER builds truncate here).
    truncated = False
    pob, ptr = _pull_optional(buf, ptr, shape_s)
    volb, ptr = _pull_optional(buf, ptr, shape_s)
    bfeldb, ptr = _pull_optional(buf, ptr, shape_s)
    bpolb, ptr  = _pull_optional(buf, ptr, shape_s)
    bradb, ptr  = _pull_optional(buf, ptr, shape_s)
    btorb, ptr  = _pull_optional(buf, ptr, shape_s)

    # Skip 8 "dummy" / future-use blocks (4 per-species + 4 scalar)
    # before the sheath-delta pair. Jeremy's reader consumes but
    # discards them. Do the same.
    for _ in range(4):
        _dummy, ptr = _pull_optional(buf, ptr, shape_n)
        if _dummy is None:
            truncated = True
            break
    if not truncated:
        for _ in range(4):
            _dummy, ptr = _pull_optional(buf, ptr, shape_s)
            if _dummy is None:
                truncated = True
                break

    delta_sheathxb = delta_sheathyb = None
    if not truncated:
        delta_sheathxb, ptr = _pull_optional(buf, ptr, shape_s)
        delta_sheathyb, ptr = _pull_optional(buf, ptr, shape_s)

    zib = None
    if not truncated:
        zib, ptr = _pull_optional(buf, ptr, shape_n)

    return Fort31Data(
        dnib=dnib, uub=uub, vvb=vvb, wwb=wwb, upb=upb,
        fnixb=fnixb, fniyb=fniyb, uudiab=uudiab, vvdiab=vvdiab,
        teb=teb, tib=tib, prb=prb, rrb=rrb,
        feixb=feixb, feiyb=feiyb, feexb=feexb, feeyb=feeyb,
        pob=pob, volb=volb,
        bfeldb=bfeldb, bpolb=bpolb, bradb=bradb, btorb=btorb,
        delta_sheathxb=delta_sheathxb, delta_sheathyb=delta_sheathyb,
        zib=zib,
        nx=nx, ny=ny, ns_plasma=ns_plasma, file=str(p),
        truncated=truncated,
    )


def _demo():
    import argparse
    ap = argparse.ArgumentParser(description="Parse a fort.31 and print block summaries.")
    ap.add_argument("run_path", type=Path)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--ns-plasma", type=int, required=True)
    args = ap.parse_args()
    ft = read_fort31(args.run_path, args.nx, args.ny, args.ns_plasma)
    print(f"fort.31: {ft.file}")
    print(f"  grid = ({ft.nx}+2) x ({ft.ny}+2),  ns_plasma = {ft.ns_plasma}")
    print(f"  truncated: {ft.truncated}")
    # Pretty-print summary stats
    items = [("dnib", ft.dnib), ("teb[eV]", ft.teb), ("tib[eV]", ft.tib),
             ("prb[Pa]", ft.prb), ("pob[V]", ft.pob), ("volb[m^3]", ft.volb),
             ("|B|[T]", ft.bfeldb), ("Bpol[T]", ft.bpolb),
             ("Brad[T]", ft.bradb), ("Btor[T]", ft.btorb)]
    for name, arr in items:
        if arr is None:
            print(f"  {name:10s}: (not present)")
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            print(f"  {name:10s}: all-NaN")
            continue
        print(f"  {name:10s}: min={finite.min():+.3e} max={finite.max():+.3e} "
              f"median={np.median(finite):+.3e}")


if __name__ == "__main__":
    _demo()

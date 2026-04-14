#!/usr/bin/env python3
"""
Generate input/plasma.h5 and input/bfield.h5 for test_west_axi from SOLEDGE3X.

Example:
  python3 generate_west_input.py \
    --base-dir /path/to/soledge/run_dir \
    --gfile /path/to/geqdsk
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent  # .../test_west_axi/analysis
CONVERTER_DIR = ROOT.parents[2] / "tools" / "converters"  # .../OpenEdge/tools/converters
sys.path.insert(0, str(CONVERTER_DIR))


def _install_matplotlib_stubs() -> None:
    class _PathStub:
        def __init__(self, vertices):
            self.vertices = np.asarray(vertices, dtype=float)

        def contains_points(self, points):
            pts = np.asarray(points, dtype=float)
            inside = np.zeros(len(pts), dtype=bool)
            poly = self.vertices
            n = len(poly) - 1
            for ip, (xp, yp) in enumerate(pts):
                flag = False
                for i in range(n):
                    x1, y1 = poly[i]
                    x2, y2 = poly[i + 1]
                    crosses = ((y1 > yp) != (y2 > yp))
                    if crosses:
                        xcross = x1 + (yp - y1) * (x2 - x1) / (y2 - y1)
                        if xp < xcross:
                            flag = not flag
                inside[ip] = flag
            return inside

    mpl_mod = types.ModuleType("matplotlib")
    mpl_path_mod = types.ModuleType("matplotlib.path")
    mpl_path_mod.Path = _PathStub
    mpl_pyplot_mod = types.ModuleType("matplotlib.pyplot")
    sys.modules.setdefault("matplotlib", mpl_mod)
    sys.modules["matplotlib.path"] = mpl_path_mod
    sys.modules["matplotlib.pyplot"] = mpl_pyplot_mod


def _install_utilities_stub() -> None:
    class _Exterior:
        def __init__(self, xy):
            self.xy = xy

    class _Polygon:
        def __init__(self, points):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if points[0] != points[-1]:
                xs.append(points[0][0])
                ys.append(points[0][1])
            self.exterior = _Exterior((xs, ys))

    class _Surface:
        def __init__(self, filename, geom_type):
            del geom_type
            points = {}
            lines = []
            section = None
            with open(filename, "r") as stream:
                for raw in stream:
                    line = raw.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if low == "points":
                        section = "points"
                        continue
                    if low == "lines":
                        section = "lines"
                        continue
                    parts = line.split()
                    if section == "points" and len(parts) >= 3 and parts[0].isdigit():
                        points[int(parts[0])] = (float(parts[1]), float(parts[2]))
                    elif section == "lines" and len(parts) >= 3 and parts[0].isdigit():
                        lines.append((int(parts[1]), int(parts[2])))
            if lines:
                chain = [lines[0][0], lines[0][1]]
                rem = lines[1:]
                while rem:
                    last = chain[-1]
                    idx = None
                    for i, (a, b) in enumerate(rem):
                        if a == last:
                            chain.append(b)
                            idx = i
                            break
                        if b == last:
                            chain.append(a)
                            idx = i
                            break
                    if idx is None or chain[-1] == chain[0]:
                        break
                    rem.pop(idx)
                point_list = [points[i] for i in chain if i in points]
            else:
                point_list = [points[i] for i in sorted(points)]
            self.polygon = _Polygon(point_list)

    utilities_mod = types.ModuleType("utilities")
    utilities_mod.surface = _Surface
    sys.modules["utilities"] = utilities_mod


try:
    import matplotlib.path  # type: ignore  # noqa: F401
    import matplotlib.pyplot  # type: ignore  # noqa: F401
except Exception:
    import numpy as np

    _install_matplotlib_stubs()

_install_utilities_stub()

from convert_s3x_plasma import interpolate_and_save_plasma_field  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True, help="SOLEDGE3X run directory containing refParam_raptorX.h5, meshEIRENE.h5, plasmaFinal.h5, mesh_raptorX.h5")
    p.add_argument("--gfile", type=Path, default=None, help="Optional GEQDSK file for B-field")
    p.add_argument("--equ-file", type=Path, default=None, help="Optional .equ file for B-field")
    p.add_argument("--nR", type=int, default=200, help="R grid size for OpenEdge plasma.h5")
    p.add_argument("--nZ", type=int, default=200, help="Z grid size for OpenEdge plasma.h5")
    p.add_argument("--main-ion-spec", type=int, default=1, help="Main ion spec index in SOLEDGE3X output")
    p.add_argument("--data-file", default=None, help="Plasma snapshot filename inside base-dir (default: plasmaFinal.h5 or latest plasma_*.h5)")
    p.add_argument("--case", default=None, help="Tag appended to debug PNG filenames (default: data-file stem)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_dir.resolve()
    if not base.exists():
        raise SystemExit(f"Base dir does not exist: {base}")

    ref_file = base / "refParam_raptorX.h5"
    mesh_file = base / "meshEIRENE.h5"
    # Data snapshot: plasmaFinal.h5 if present, otherwise a user-supplied
    # --data-file, otherwise the newest plasma_*.h5 in base.
    if args.data_file:
        data_file = base / args.data_file
    elif (base / "plasmaFinal.h5").exists():
        data_file = base / "plasmaFinal.h5"
    else:
        candidates = sorted(base.glob("plasma_*.h5"))
        if not candidates:
            raise SystemExit(f"No plasma snapshot found in {base}")
        data_file = candidates[-1]
    bfield_file = base / "mesh_raptorX.h5"
    config_file = base / "mesh.h5"
    required = [ref_file, mesh_file, data_file, bfield_file]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required SOLEDGE3X files:\n" + "\n".join(str(p) for p in missing))

    input_dir = ROOT.parent / "input"  # .../test_west_axi/input
    input_dir.mkdir(parents=True, exist_ok=True)

    plasma_out = input_dir / "plasma.h5"
    wall_out = input_dir / "wall.txt"
    wall_flux_csv = input_dir / "vv_values.csv"
    case_tag = args.case or data_file.stem
    debug_plot = input_dir / f"soledge_fields_{case_tag}.png"
    flux_total_plot = input_dir / f"soledge_flux_total_{case_tag}.png"
    flux_species_plot = input_dir / f"soledge_flux_species_{case_tag}.png"
    wall_flux_plot = input_dir / f"soledge_flux_wallcoord_{case_tag}.png"

    interpolate_and_save_plasma_field(
        str(ref_file),
        str(mesh_file),
        str(bfield_file),
        str(data_file),
        None,
        str(plasma_out),
        None,
        gfile=str(args.gfile.resolve()) if args.gfile else None,
        equ_file=str(args.equ_file.resolve()) if args.equ_file else None,
        debug_plot_file=str(debug_plot),
        flux_total_plot_file=str(flux_total_plot),
        flux_species_plot_file=str(flux_species_plot),
        wall_flux_plot_file=str(wall_flux_plot),
        wall_flux_csv_file=str(wall_flux_csv),
        nR=args.nR,
        nZ=args.nZ,
        main_ion_spec=args.main_ion_spec,
        use_mesh_wall=True,
        wall_sparta_file=str(wall_out),
        core_sparta_file=None,
        core_psi_level=None,
        config_file=str(config_file) if config_file.exists() else None,
    )

    print(f"Wrote {plasma_out}")
    print(f"Wrote {wall_out}")
    print(f"Used data snapshot: {data_file}")


if __name__ == "__main__":
    main()

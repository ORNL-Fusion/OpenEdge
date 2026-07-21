#!/usr/bin/env python3
"""Subdivide long line segments of a SPARTA 2D surf file.

Splits every segment longer than --maxlen into equal collinear pieces.
Optional --region zlo zhi rlo rhi (repeatable) restricts splitting to
segments whose midpoint falls inside a box, with its own --region-maxlen.
Geometry is unchanged: sub-segments reuse the original endpoints exactly
and keep the walk order, so winding (and read_surf invert) is preserved.

Usage:
  python3 subdivide_surf.py input/wall.surf input/wall_fine.surf --maxlen 0.01
"""

import argparse
import math
import sys


def read_surf(path):
    pts, lines = {}, []
    sec = None
    for raw in open(path):
        t = raw.split()
        if not t or t[0] == '#':
            continue
        if t[0] == 'Points':
            sec = 'P'
            continue
        if t[0] == 'Lines':
            sec = 'L'
            continue
        if sec == 'P' and len(t) >= 3:
            pts[int(t[0])] = (float(t[1]), float(t[2]))
        elif sec == 'L' and len(t) >= 3:
            lines.append((int(t[1]), int(t[2])))
    return pts, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--maxlen', type=float, default=0.01,
                    help='max segment length everywhere (m)')
    ap.add_argument('--region', nargs=5, action='append', default=[],
                    metavar=('ZLO', 'ZHI', 'RLO', 'RHI', 'MAXLEN'),
                    help='zlo zhi rlo rhi maxlen: finer limit inside a box')
    args = ap.parse_args()

    pts, lines = read_surf(args.infile)

    # Walk lines in order; emit points/lines with fresh consecutive ids.
    # Points shared between consecutive segments keep one id so the loop
    # stays watertight.
    new_pts = []          # (x, y)
    pt_id = {}            # original point id -> new id (originals only)

    def add_orig(pid):
        if pid not in pt_id:
            new_pts.append(pts[pid])
            pt_id[pid] = len(new_pts)
        return pt_id[pid]

    new_lines = []
    nsplit = 0
    for a, b in lines:
        (x1, y1), (x2, y2) = pts[a], pts[b]
        L = math.hypot(x2 - x1, y2 - y1)
        zm, rm = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        lim = args.maxlen
        for zlo, zhi, rlo, rhi, ml in ((float(v) for v in r) for r in args.region):
            if zlo <= zm <= zhi and rlo <= rm <= rhi:
                lim = min(lim, ml)
        n = max(1, math.ceil(L / lim - 1e-9))
        ia = add_orig(a)
        prev = ia
        for k in range(1, n):
            t = k / n
            new_pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            cur = len(new_pts)
            new_lines.append((prev, cur))
            prev = cur
        ib = add_orig(b)
        new_lines.append((prev, ib))
        if n > 1:
            nsplit += 1

    with open(args.outfile, 'w') as f:
        f.write('surface geometry: %s subdivided, maxlen %g\n\n'
                % (args.infile, args.maxlen))
        f.write('%d points\n%d lines\n\nPoints\n\n' % (len(new_pts), len(new_lines)))
        for i, (x, y) in enumerate(new_pts, 1):
            f.write('%d %.10g %.10g\n' % (i, x, y))
        f.write('\nLines\n\n')
        for i, (a, b) in enumerate(new_lines, 1):
            f.write('%d %d %d\n' % (i, a, b))

    print('%s: %d points / %d lines  ->  %s: %d points / %d lines '
          '(%d segments split)'
          % (args.infile, len(pts), len(lines),
             args.outfile, len(new_pts), len(new_lines), nsplit))


if __name__ == '__main__':
    sys.exit(main())

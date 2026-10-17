#!/usr/bin/env python3
"""Side-by-side video: VTK render of the simulated grains and Li species next to the
ST40 fast-camera frame at the same shot time.

    python scripts/make_camera_video.py output --video ~/Downloads/st40_video/st40_plasma_in_colour_youtube.mp4

Camera: at (R, phi, Z) looking at (0, 0, look_Z); phi in the simulation frame (dropper
at phi = 0). shot_ms = sim_ms - t_offset; video_s = (shot_ms - STAMP_B) / STAMP_A from
the overlay time stamps. States are read from state.*.vtu or a single text dump `state`.
Colours follow the footage: powder red-orange, Li I red, Li II green; Li III/IV hidden.
Li I / Li II brightness ~ n_e exp(-E_line/Te) at the particle (a line-emission proxy)."""
import argparse, glob, re, subprocess, tempfile
from pathlib import Path
import numpy as np
import h5py
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CASE = Path(__file__).resolve().parents[1]
DT = 2.0e-6
STAMP_A, STAMP_B = 4.6875, -4.6
SPECIES = {1: ("#ff6a3d", 5, 1.0), 2: ("#e01030", 4, 0.45), 3: ("#30ff50", 4, 0.45), 4: ("#3080ff", 3, 0.25)}
E_LINE = {2: 1.85, 3: 61.0}        # eV: Li I 670.8 nm, Li II 548.5 nm upper levels


class Plasma:
    """Te, ne at (R, Z) from the mesh of the plasma file (nearest triangle centre)."""
    def __init__(self, path):
        g = h5py.File(path, "r")["mesh"]
        tri = g["triangles"][...]; vr = g["vtx_r"][...]; vz = g["vtx_z"][...]; cidx = g["cell_index"][...]
        ok = (cidx >= 0) & (cidx < len(g["temp_e"]))
        self.tree = cKDTree(np.column_stack([vr[tri].mean(1)[ok], vz[tri].mean(1)[ok]])); self.cid = cidx[ok]
        self.te, self.ne = g["temp_e"][...], g["dens_e"][...]

    def emission(self, P, E_eV):
        """n_e exp(-E/Te) at the particle positions, normalised to its 99th percentile."""
        d, i = self.tree.query(np.column_stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]]))
        te = np.maximum(self.te[self.cid[i]], 1e-3); w = self.ne[self.cid[i]] * np.exp(-E_eV / te)
        w[d > 0.03] = 0.0
        top = np.percentile(w, 99) if (w > 0).any() else 1.0
        return np.clip(w / max(top, 1e-300), 0.0, 1.0)


def load_states(run):
    files = sorted(glob.glob(str(run / "state.*.vtu")), key=lambda p: int(re.findall(r"\d+", p)[-1]))
    if files:
        for f in files:
            m = pv.read(f); yield int(re.findall(r"\d+", f)[-1]), np.asarray(m.points), np.asarray(m["type"])
        return
    step = hdr = None; rows = []
    def flush():
        A = np.array(rows); return step, A[:, [hdr.index("x"), hdr.index("y"), hdr.index("z")]], A[:, hdr.index("type")].astype(int)
    for line in open(run / "state"):
        t = line.split()
        if t[:2] == ["ITEM:", "TIMESTEP"]:
            if rows: yield flush()
            step = hdr = None; rows = []; continue
        if step is None and len(t) == 1 and t[0].isdigit(): step = int(t[0]); continue
        if t[:2] == ["ITEM:", "ATOMS"]: hdr = t[2:]; continue
        if hdr and len(t) == len(hdr): rows.append([float(v) for v in t])
    if rows: yield flush()


def video_frame(video, shot_ms, out_png):
    s = (shot_ms - STAMP_B) / STAMP_A
    if s < 0: return None
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "crop=iw*0.52:ih:iw*0.24:0", str(out_png)], capture_output=True)
    return out_png if r.returncode == 0 and Path(out_png).exists() else None


def render(P, typ, wall, cam, show_stripped, plasma, size=800):
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("black")
    pl.add_mesh(wall, color="#9aa0a8", opacity=0.07, lighting=False)
    pl.add_mesh(pv.Sphere(radius=0.012, center=(0.51, 0.0, 0.845)), color="yellow", lighting=False)
    for sp, (col, ps, op) in SPECIES.items():
        if sp == 4 and not show_stripped: continue
        sel = (typ >= 4) if sp == 4 else (typ == sp)
        if not sel.any(): continue
        pts = P[sel]
        if plasma is not None and sp in E_LINE:
            w = plasma.emission(pts, E_LINE[sp]); keep = w > 0.02
            if not keep.any(): continue
            cloud = pv.PolyData(pts[keep]); cloud["w"] = w[keep]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("e", ["black", col])
            pl.add_mesh(cloud, scalars="w", cmap=cmap, clim=(0, 1), point_size=ps, opacity=op, render_points_as_spheres=True,
                        ambient=1.0, diffuse=0.0, show_scalar_bar=False)
        else:
            pl.add_points(pts, color=col, point_size=ps, opacity=op, render_points_as_spheres=True, ambient=1.0, diffuse=0.0)
    pl.camera.position = cam["pos"]; pl.camera.focal_point = cam["focal"]; pl.camera.up = (0, 0, 1); pl.camera.view_angle = cam["fov"]
    img = pl.screenshot(return_img=True); pl.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="directory with state.*.vtu or a text dump `state`")
    ap.add_argument("--video", default=str(Path.home() / "Downloads/st40_video/st40_plasma_in_colour_youtube.mp4"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cam-R", type=float, default=0.95)
    ap.add_argument("--cam-phi", type=float, default=-45.0)
    ap.add_argument("--cam-Z", type=float, default=0.0)
    ap.add_argument("--look-Z", type=float, default=0.35)
    ap.add_argument("--fov", type=float, default=95.0)
    ap.add_argument("--t-offset-ms", type=float, default=88.0)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--show-stripped", action="store_true", help="also draw Li2+/Li3+")
    ap.add_argument("--plasma", default=str(CASE / "input/plasma_farsol.h5"))
    ap.add_argument("--no-emission", action="store_true", help="equal brightness per particle instead of n_e exp(-E/Te)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--mirror", action="store_true", help="flip the rendered view left-right (periscope optics)")
    ap.add_argument("--top", action="store_true", help="view from above instead of the camera port")
    a = ap.parse_args()
    run = Path(a.run); out = Path(a.out) if a.out else CASE / "sim_vs_camera.mp4"
    p = np.radians(a.cam_phi)
    cam = dict(pos=(a.cam_R * np.cos(p), a.cam_R * np.sin(p), a.cam_Z), focal=(0.0, 0.0, a.look_Z), fov=a.fov)
    if a.top: cam = dict(pos=(0.0, 0.0, 3.0), focal=(0.0, 0.0, 0.0), fov=45.0)
    wall = pv.read(CASE / "input/st40_wall_3d.stl")
    plasma = None if a.no_emission else Plasma(a.plasma)
    states = list(load_states(run))
    if a.max_frames: states = states[:a.max_frames]
    tmp = Path(tempfile.mkdtemp())
    for i, (step, P, typ) in enumerate(states):
        sim_ms = step * a.dt * 1e3; shot_ms = sim_ms - a.t_offset_ms
        img = render(P, typ, wall, cam, a.show_stripped, plasma)
        if a.mirror: img = img[:, ::-1]
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
        ax[0].imshow(img); ax[0].axis("off")
        n = {sp: int((typ == sp).sum()) for sp in (1, 2, 3)}
        ax[0].set_title(f"OpenEdge   sim t = {sim_ms:.0f} ms    powder {n[1]}   Li I {n[2]}   Li II {n[3]}", color="w", fontsize=11)
        ax[1].set_facecolor("black"); ax[1].axis("off")
        png = video_frame(a.video, shot_ms, tmp / "cam.png")
        if png is not None: ax[1].imshow(mpimg.imread(png))
        ax[1].set_title(f"ST40 shot 14448 camera    shot t = {shot_ms:+.0f} ms", color="w", fontsize=11)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.02)
        fig.savefig(tmp / f"frame_{i:03d}.png", dpi=100, facecolor="black"); plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", str(tmp / "frame_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    print(f"{len(states)} frames -> {out}")


if __name__ == "__main__":
    main()

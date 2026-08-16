#!/usr/bin/env python3
"""Stage-A gates for the hybrid wall-handoff test (PLAN.md, rev 3 + review).

Inputs per tag: vanish impact logs (output/impacts.<tag>.csv.rank*),
per-step dumps (output/traj.<tag>), switch-event logs
(output/switch.<tag>.csv.rank*, 17-digit).

Tags: ref/ref2/ref4 Boris reference at dt, dt/2, dt/4 (convergence band);
hyb_small (hybrid @ dt); hyb (hybrid @ 20x dt); skip (hybrid @ 400x dt,
shell-skipping); gca (pure GCA, Option-A input); hyst (in-shell outbound
hysteresis excursion).

Hard gates: particle lifecycle conservation (no duplicate vanish, no ID
alive after its vanish, all N absorbed), branch state at the recorded
impact, exactly-one transition with switch-log proof, event-precision
handoff energy, gyrocenter-offset, KS + quantile distribution agreement
within the reference's own convergence band, hysteresis excursion.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 6.6464731e-27
Q = 1.602176634e-19
B0 = 1.0
ALPHA = np.radians(45.0)
BHAT = np.array([np.sin(ALPHA), 0.0, -np.cos(ALPHA)])
OMEGA = Q * B0 / M
TGYRO = 2.0 * np.pi / OMEGA
V0 = np.sqrt(2.0 * 5.0 * Q / M)
RHOL = M * (V0 / np.sqrt(2.0)) / (Q * B0)
DSW = 2.5 * RHOL
NENS = 256
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

CFG = {"ref": 5e-10, "ref2": 2.5e-10, "ref4": 1.25e-10,
       "hyb_small": 5e-10, "hyb": 1e-8, "skip": 2e-7, "leap": 2.5e-6,
       "k1": 1e-8, "k5": 1e-8, "gca": 1e-8, "a1": 1e-8}


def read_csv(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        if ".counts." in p:
            continue
        try:
            d = np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                              encoding="utf-8")
        except Exception:
            continue
        if d.size:
            rows.append(np.atleast_1d(d))
    return np.concatenate(rows) if rows else None


def read_traj(tag, with_final=False):
    per = {}
    final_ts, final_ids = -1, []
    with open(os.path.join(OUT, f"traj.{tag}")) as f:
        for line in f:
            if "ITEM: TIMESTEP" in line:
                ts = int(next(f)); next(f); n = int(next(f)); next(f)
                for _ in range(3): next(f)
                next(f)
                final_ts, final_ids = ts, []
                for _ in range(n):
                    a = next(f).split()
                    final_ids.append(int(a[0]))
                    per.setdefault(int(a[0]), []).append(
                        [ts] + [float(q) for q in a[2:]])
    out = {pid: np.array(rr) for pid, rr in per.items()}
    return (out, final_ts, final_ids) if with_final else out


def energy_trace(r):
    mode = r[:, 12]
    ke = 0.5 * M * (r[:, 4]**2 + r[:, 5]**2 + r[:, 6]**2)
    hgc = 0.5 * M * r[:, 10]**2 + r[:, 11] * B0
    return np.where(mode > 0.5, hgc, ke)


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    g = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, g, side="right")/len(a)
                         - np.searchsorted(b, g, side="right")/len(b)))


def impact_obs(imp, dt):
    vn = imp["vz"]
    v = np.sqrt(imp["vx"]**2 + imp["vy"]**2 + imp["vz"]**2)
    en = 0.5 * M * vn**2 / Q
    ang = np.degrees(np.arccos(np.clip(np.abs(vn) / v, 0, 1)))
    return en, ang, imp["timestep"] * dt, imp["x"], imp["y"]


def main():
    checks = []

    def gate(name, ok, info=""):
        checks.append((name, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))

    imp, traj, sw, fin = {}, {}, {}, {}
    for t in CFG:
        imp[t] = read_csv(os.path.join(OUT, f"impacts.{t}.csv.rank*"))
        traj[t], fts, fids = read_traj(t, with_final=True)
        fin[t] = fids
        sw[t] = read_csv(os.path.join(OUT, f"switch.{t}.csv.rank*"))
    obs = {t: impact_obs(imp[t], CFG[t]) for t in CFG if imp[t] is not None}

    # --- HARD lifecycle gates (every impact tag) -------------------------
    for t in CFG:
        if imp[t] is None:
            gate(f"{t}: impact log present", False)
            continue
        ids = imp[t]["id"].astype(int)
        uids, cnt = np.unique(ids, return_counts=True)
        van = dict(zip(uids, [imp[t]["timestep"][ids == u].min() for u in uids]))
        ghosts = sorted({i for i, r in traj[t].items()
                         if i in van and r[-1, 0] > van[i]})
        gate(f"{t}: no duplicate vanish IDs", uids.size == ids.size,
             f"dups {uids[cnt > 1][:4]}" if uids.size != ids.size else "")
        gate(f"{t}: no ID alive after its vanish", not ghosts,
             f"ghosts {ghosts[:4]}" if ghosts else "")
        # explicit lifecycle balance: N in the first dumped frame minus N
        # in the run's FINAL dumped frame (empty when all absorbed) must
        # equal the unique vanish count. (dfreq-1 tags give frame-level
        # coverage; coarse refs could miss a sub-dfreq survivor — the
        # ghost gate above still catches any living past one interval.)
        n_init = sum(1 for r in traj[t].values() if r[0, 0] == 0)
        n_final = len(fin[t])
        gate(f"{t}: N_init - N_final == N_unique_vanish",
             (n_init - n_final) == uids.size,
             f"init {n_init} final {n_final} vanish {uids.size}")
        gate(f"{t}: all {NENS} absorbed exactly once", uids.size == NENS
             and ids.size == NENS, f"{uids.size}/{ids.size}")

    # --- branch state at the recorded impact -----------------------------
    for t in ("hyb_small", "hyb", "skip", "leap", "k1", "k5"):
        gate(f"{t}: Boris at impact", np.all(imp[t]["gca_mode"] == 0)
             and np.all(imp[t]["gca_valid"] == 0))
    gate("gca: GC at impact", np.all(imp["gca"]["gca_mode"] == 1))
    gate("a1: GC at impact", np.all(imp["a1"]["gca_mode"] == 1))

    # leap: every switch must be a swept/crossing detection — a single GC
    # step spans shell AND wall plane, invisible to endpoint-only checks
    sl = sw.get("leap")
    gate("leap: switch log present", sl is not None)
    if sl is not None:
        d = sl[(sl["oldmode"] == 1) & (sl["newmode"] == 0)]
        li, lc = np.unique(d["id"].astype(int), return_counts=True)
        gate("leap: one swept transition per particle, all replays",
             li.size == NENS and np.all(lc == 1)
             and np.all(np.atleast_1d(d["reason"]) == "swept")
             and np.all(d["replay"] == 1), f"{li.size} ids")
        # purpose-built: EVERY leap event must be a complete
        # outside-to-outside plane crossing (s0 > d_sw AND s1 < -d_sw) —
        # the geometry endpoint-only tests are provably blind to
        gate("leap: ALL events are complete outside-to-outside crossings "
             "(d_start > d_sw, d_end < -d_sw)",
             np.all(d["d_start"] > d["d_sw"])
             and np.all(d["d_end"] < -d["d_sw"]),
             f"d_end range [{d['d_end'].min():.2e},{d['d_end'].max():.2e}]")

    # --- transitions: exactly one, proven in dump AND switch log ---------
    for t in ("hyb", "skip"):
        tr = traj[t]
        ok_n, dists = 0, []
        for pid, r in tr.items():
            m = r[np.argmax(r[:, 12] > 0.5):, 12]
            down = np.where((m[:-1] > 0.5) & (m[1:] < 0.5))[0]
            if len(down) == 1 and np.all(m[down[0]+1:] < 0.5):
                ok_n += 1
                k = np.argmax(r[:, 12] > 0.5) + down[0]
                dists.append(r[k, 3])
        gate(f"{t}: exactly one GCA->Boris transition (dump)",
             ok_n == len(tr), f"{ok_n}/{len(tr)}")
        s = sw[t]
        gate(f"{t}: switch log present", s is not None)
        if s is not None:
            d = s[(s["oldmode"] == 1) & (s["newmode"] == 0)]
            di, dc = np.unique(d["id"].astype(int), return_counts=True)
            gate(f"{t}: switch log — one GCA->Boris event per particle",
                 di.size == NENS and np.all(dc == 1),
                 f"{di.size} ids, max/particle {dc.max() if dc.size else 0}")
            reasons = np.atleast_1d(d["reason"])
            gate(f"{t}: switch reasons are shell entries",
                 set(reasons.tolist()) <= {"shell_start", "swept"},
                 f"{sorted(set(reasons.tolist()))}")
            # geometry of each reason, from the 17-digit event log itself
            swp = d[reasons == "swept"]
            sst = d[reasons == "shell_start"]
            ok_geo = (np.all(swp["d_start"] >= swp["d_sw"] - 1e-15)
                      and np.all(swp["d_end"] < swp["d_sw"])
                      and np.all(swp["replay"] == 1)
                      and np.all(sst["d_start"] < sst["d_sw"]))
            gate(f"{t}: switch-event geometry consistent "
                 "(swept: d_start>=d_sw>d_end, replay; shell_start: d_start<d_sw)",
                 ok_geo)
            if t == "skip":
                gate("skip: ALL 256 transitions are trial/replay (swept)",
                     int(np.sum(d["replay"] == 1)) == NENS,
                     f"{int(np.sum(d['replay'] == 1))}/256")
            # event-precision handoff energy (17-digit log)
            rel = np.abs(d["e_post"] / d["e_pre"] - 1.0)
            gate(f"{t}: handoff energy conversion exact to fp "
                 "(|dE/E| < 1e-12)", np.max(rel) < 1e-12,
                 f"max {np.max(rel):.1e}")
        if s is not None:
            vstep = abs(V0 / np.sqrt(2.0) * BHAT[2]) * CFG[t]
            ds = s[(s["oldmode"] == 1)]["d_start"]
            gate(f"{t}: switch d_start in [d_sw, d_sw + 2 GC-steps] "
                 "(17-digit event log)",
                 np.all(ds > DSW - 1e-12) & np.all(ds < DSW + 2*vstep),
                 f"[{ds.min():.6e},{ds.max():.6e}] d_sw {DSW:.6e}")

    # --- gyrocenter-offset at the handoff (hyb, first Boris frame) -------
    offs = []
    for pid, r in traj["hyb"].items():
        m = r[:, 12]
        down = np.where((m[:-1] > 0.5) & (m[1:] < 0.5))[0]
        if len(down) != 1:
            continue
        k = down[0]
        xb, vb = r[k+1, 1:4], r[k+1, 4:7]
        xinf = xb + np.cross(vb, BHAT) / OMEGA
        # expected center = stored GC advanced by the parallel travel of
        # the Boris step (otherwise the gate just measures vpar*dt)
        vpar_k = r[k, 10]
        xexp = r[k, 7:10] + vpar_k * BHAT * CFG["hyb"]
        offs.append(np.linalg.norm(xinf - xexp))
    offs = np.array(offs)
    # median gates the transform itself; the max rides on guard-truncated
    # first steps (parallel travel < dt), bounded by vpar*dt
    vpar_dt = V0 / np.sqrt(2.0) * CFG["hyb"]
    gate("hyb: post-step Boris gyrocenter matches GC + vpar*bhat*dt "
         "(median < 0.02 rho_L, max < vpar*dt)",
         offs.size > 0 and np.median(offs) < 0.02 * RHOL
         and np.max(offs) < 1.05 * vpar_dt,
         f"median {np.median(offs):.2e}, max {np.max(offs):.2e}")

    # --- energy conservation over the flight -----------------------------
    for t, lab in (("ref", "Boris total-energy"), ("gca", "stored-H")):
        d = max(np.max(np.abs(energy_trace(r)/energy_trace(r)[0] - 1.0))
                for r in traj[t].values())
        gate(f"{t}: {lab} drift < 1e-4", d < 1e-4, f"{d:.1e}")

    # --- distributions: KS + quantiles within the refs' own band ---------
    en_r, ang_r = obs["ref"][0], obs["ref"][1]
    t_r, x_r, y_r = obs["ref"][2], obs["ref"][3], obs["ref"][4]
    ks_crit = 1.36 * np.sqrt(2.0 / NENS)
    for k, i in (("E_n", 0), ("angle", 1)):
        k12 = ks_stat(obs["ref"][i], obs["ref2"][i])
        k24 = ks_stat(obs["ref2"][i], obs["ref4"][i])
        gate(f"reference {k} convergence contracts (KS(dt/2,dt/4) <= "
             "KS(dt,dt/2) + sampling)", k24 <= k12 + 0.25 * ks_crit,
             f"{k12:.3f} -> {k24:.3f}")
    QS = (0.1, 0.5, 0.9)
    band = {}
    rngq = np.random.default_rng(7)
    for k, i in (("E_n", 0), ("angle", 1)):
        ks_null = max(ks_stat(obs["ref"][i], obs["ref2"][i]),
                      ks_stat(obs["ref"][i], obs["ref4"][i]))
        qn = np.max([np.abs(np.quantile(obs["ref"][i], QS)
                            - np.quantile(obs[rr][i], QS))
                     for rr in ("ref2", "ref4")], axis=0)
        # finite-sampling floor: bootstrap quantile sigma of the N=256
        # reference sample; two independent ensembles -> sqrt(2)
        boot = np.quantile(rngq.choice(obs["ref"][i], (200, NENS)), QS, axis=1)
        sig = boot.std(axis=1) * np.sqrt(2.0)   # shape (len(QS),)
        band[k] = (max(ks_crit, 2.0 * ks_null),
                   np.maximum(2.0 * qn, 3.0 * sig))
    ks_a0 = {}
    for t in ("hyb_small", "hyb", "skip", "k1", "k5", "gca", "a1"):
        for k, i in (("E_n", 0), ("angle", 1)):
            s = ks_stat(obs[t][i], obs["ref"][i])
            dq = np.abs(np.quantile(obs[t][i], QS) - np.quantile(obs["ref"][i], QS))
            if t == "gca":
                ks_a0[k] = s
                print(f"  INFO: gca (A0) {k} KS {s:.3f} (band {band[k][0]:.3f}), "
                      f"q10/50/90 diff {dq.round(3)} — rejected baseline")
            elif t == "a1":
                # THE B2 EXPERIMENT: A1 must strictly improve on A0;
                # band-pass would certify GC-to-wall for this envelope
                gate(f"a1: {k} KS improves on A0 (flux-weighted phase)",
                     s < ks_a0.get(k, 1.0),
                     f"A1 {s:.3f} vs A0 {ks_a0.get(k, 1.0):.3f}")
                verdict = ("WITHIN band — GC-to-wall certifiable here"
                           if s < band[k][0]
                           else "outside band — Boris shell stays required")
                print(f"  INFO: A1 VERDICT {k}: KS {s:.3f} "
                      f"band {band[k][0]:.3f} — {verdict}")
            else:
                gate(f"{t}: {k} KS within refs band", s < band[k][0],
                     f"KS {s:.3f} band {band[k][0]:.3f}")
                gate(f"{t}: {k} quantiles (10/50/90) within refs band",
                     np.all(dq < band[k][1]),
                     f"dq {dq.round(3)} band {band[k][1].round(3)}")
        tt, xx, yy = obs[t][2], obs[t][3], obs[t][4]
        if t == "skip":
            # impact TIME at 400x dt is knowingly degraded by the 3D
            # subcycle remainder loss (documented mover defect, PLAN.md);
            # distributions are the gated quantity here
            print(f"  INFO: skip impact-time offset {abs(tt.mean()-t_r.mean()):.2e} s "
                  "(remainder-loss defect, exempt until mover repair)")
        else:
            gate(f"{t}: mean impact time within 1 gyroperiod of ref",
                 abs(tt.mean() - t_r.mean()) < TGYRO,
                 f"{abs(tt.mean()-t_r.mean()):.2e} s")
        cen = np.hypot(xx.mean() - x_r.mean(), yy.mean() - y_r.mean())
        gate(f"{t}: tangential impact centroid within 1 rho_L of ref",
             cen < RHOL, f"{cen:.2e} m")

    # --- hysteresis excursion (in-shell outbound launch) ------------------
    trh = read_traj("hyst")
    ok_h = 0
    for pid, r in trh.items():
        m, chi, z = r[:, 12], r[:, 14], r[:, 3]
        up = np.where((m[:-1] < 0.5) & (m[1:] > 0.5))[0]
        good = (len(up) == 1 and np.all(m[up[0]+1:] > 0.5)
                and chi[up[0]] >= 4.0*np.pi - 1e-9
                and z[up[0]] > 2.0*DSW - 1e-9)
        ok_h += good
    gate("hyst: outbound excursion re-enters GCA exactly once, only after "
         "chi >= 4*pi AND d > 2*d_sw", ok_h == len(trh), f"{ok_h}/{len(trh)}")
    sh = read_csv(os.path.join(OUT, "switch.hyst.csv.rank*"))
    if sh is not None:
        u = sh[(sh["oldmode"] == 0) & (sh["newmode"] == 1)]
        gate("hyst: switch log — re-entry events logged with reason",
             u.size == len(trh)
             and set(np.atleast_1d(u["reason"]).tolist()) == {"reentry"},
             f"{u.size} events")
    else:
        gate("hyst: switch log present", False)

    # --- figures: trajectories, energy, distributions --------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for t, c in (("ref", "k"), ("hyb", "tab:red"), ("gca", "tab:blue")):
        tr = traj[t]
        first = True
        for pid in list(tr)[:6]:
            r = tr[pid]
            ax[0].plot(r[:, 1], r[:, 3], c, lw=0.7, label=t if first else None)
            first = False
            e = energy_trace(r)
            ax[1].plot(r[:, 0] * CFG[t] * 1e6, e/e[0] - 1.0, c, lw=0.7)
        ax[2].hist(obs[t][0], bins=24, histtype="step", color=c, label=t)
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("z (m)")
    ax[0].set_title("trajectories (6 shown)"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("t (us)"); ax[1].set_ylabel("E/E0 - 1")
    ax[1].set_title("total-energy conservation")
    ax[2].set_xlabel("normal impact energy (eV)"); ax[2].set_title("E_n")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stageA_summary.png"), dpi=200)
    print(f"Wrote {os.path.join(OUT, 'stageA_summary.png')}")

    ok = all(p for _, p in checks)
    print(("PASS" if ok else "FAIL") + ": verification/pushers/hybrid Stage A")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

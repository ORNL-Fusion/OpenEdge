#!/usr/bin/env python3
"""Stage a self-contained coupling campaign so the back-and-forth loop never
mutates the validated standalone inputs.

Copies the chosen case's SOLPS dir + the OpenEdge deck/plasma/surface into a
fresh campaign tree, rewrites the deck's baked-in absolute paths to point at
the campaign, resets the SOLPS case to its pre-coupling state, and emits a
campaign config.json that loop_driver runs against. The canonical
coupled_test inputs are left untouched (baserun, wall.surf, and the run
scripts are shared read-only via symlink / absolute path).

    campaigns/<case>/
        solps/<case>/        copy of the case (regenerated outputs + stale
                             source2d/she2d/b2.sources.profile stripped;
                             b2mn.dat reset from .openedge.bak)
        solps/baserun  ->    symlink to canonical baserun (shared, read-only)
        openedge/
            full_runs/<case>/in.oedge_wall   (paths rewritten to campaign)
            plasma/, surf_with_strip/, strip_csv/, runs/
        config.json          loop_driver entry point

    python3 setup_coupling.py config.json --case attached__mist [--fresh] [--dry]
"""
import os
import sys
import json
import shutil
import argparse

# SOLPS outputs/artifacts that must NOT carry into a fresh campaign: either
# regenerated each run, or left over from a previous coupling.
SOLPS_SKIP = shutil.ignore_patterns(
    "b2mn.exe.dir", "b2fplasma", "b2fplasmf", "b2fmovie", "b2time.nc",
    "b2tallies.nc", "b2ftrace", "b2ftrack", "run.log", "*.last10", "*~",
    ".quit", ".pause", "source2d.*", "she2d.*", "b2.sources.profile",
    "b2mn.dat.openedge.bak", "b2pl.exe.dir",
)

# OpenEdge files to stage from full_runs/<case>/ (deck + species).
OE_CASE_FILES = ("in.oedge_wall", "droplet.species", "li.species")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--case", required=True)
    ap.add_argument("--campaign-root", default=None,
                    help="default <coupled>/campaigns")
    ap.add_argument("--fresh", action="store_true", help="rebuild even if it exists")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="stage even if the source SOLPS run did not finish")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    coupled = cfg["coupled_dir"]
    case = args.case
    base = case.split("__")[0]
    cc = cfg["cases"][case]

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    camp_root = args.campaign_root or os.path.join(coupled, "campaigns")
    campaign = os.path.join(camp_root, case)
    src_solps = absp(cc["solps_run_dir"])
    src_deck = absp(cc["oe_deck"])
    src_deck_dir = os.path.dirname(src_deck)
    baserun = absp(cfg.get("solps_base_dir", "solps/baserun"))

    plasma_src = absp(os.path.join("openedge", "plasma", f"plasma_{base}.h5"))
    surf_dir_src = absp("openedge/surf_with_strip")
    strip_dir_src = absp("openedge/strip_csv")

    print(f"== stage coupling campaign  case='{case}' ==")
    print(f"  source SOLPS : {src_solps}")
    print(f"  source deck  : {src_deck}")
    print(f"  baserun (ro) : {baserun}")
    print(f"  campaign     : {campaign}")

    # Refuse to stage from a SOLPS run that never completed (no b2fplasma) --
    # its converged state is invalid, so the coupling would start from garbage.
    def _nonempty(p):
        return os.path.exists(p) and os.path.getsize(p) > 0
    finished = (_nonempty(os.path.join(src_solps, "b2fplasma")) or
                _nonempty(os.path.join(src_solps, "b2fplasmf"))) and \
        (_nonempty(os.path.join(src_solps, "b2fstate")) or
         _nonempty(os.path.join(src_solps, "b2fstati")))
    if not finished and not args.allow_incomplete:
        sys.exit(f"ERROR: source SOLPS run looks unfinished (no b2fplasma or no "
                 f"usable state) in\n  {src_solps}\n"
                 f"  rerun SOLPS to completion, or pass --allow-incomplete to "
                 f"stage anyway.")

    if os.path.exists(campaign):
        if not args.fresh:
            sys.exit(f"campaign exists: {campaign}\n  re-run with --fresh to rebuild it.")
        print(f"  --fresh: removing existing {campaign}")
        if not args.dry:
            shutil.rmtree(campaign)

    # campaign layout
    camp_solps = os.path.join(campaign, "solps", case)
    camp_oe = os.path.join(campaign, "openedge")
    camp_deck_dir = os.path.join(camp_oe, "full_runs", case)
    dirs = [os.path.dirname(camp_solps), camp_deck_dir,
            os.path.join(camp_oe, "plasma"),
            os.path.join(camp_oe, "surf_with_strip"),
            os.path.join(camp_oe, "strip_csv"),
            os.path.join(camp_oe, "runs")]

    if args.dry:
        print("\n--dry: would create campaign tree, copy SOLPS case + OpenEdge "
              "inputs, rewrite deck paths, and write config.json. Nothing done.")
        return

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. SOLPS case (lean copy) + shared baserun symlink + reset b2mn.dat
    shutil.copytree(src_solps, camp_solps, ignore=SOLPS_SKIP, dirs_exist_ok=True)
    os.symlink(os.path.abspath(baserun), os.path.join(campaign, "solps", "baserun"))
    bak = os.path.join(src_solps, "b2mn.dat.openedge.bak")
    if os.path.exists(bak):
        shutil.copy2(bak, os.path.join(camp_solps, "b2mn.dat"))
        print("  reset b2mn.dat from .openedge.bak (pre-coupling state)")

    # 2. OpenEdge inputs
    shutil.copy2(plasma_src, os.path.join(camp_oe, "plasma", f"plasma_{base}.h5"))
    for name in ("wall_refined.surf", f"wall_strip_{base}.surf"):
        src = os.path.join(surf_dir_src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(camp_oe, "surf_with_strip", name))
    for side in ("outer", "inner"):
        src = os.path.join(strip_dir_src, f"{base}.strip.{side}.csv")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(camp_oe, "strip_csv", os.path.basename(src)))
    for name in OE_CASE_FILES:
        src = os.path.join(src_deck_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(camp_deck_dir, name))

    # 3. rewrite the deck's baked absolute openedge paths -> campaign
    deck_path = os.path.join(camp_deck_dir, "in.oedge_wall")
    with open(deck_path) as f:
        deck = f.read()
    deck = deck.replace(os.path.join(coupled, "openedge"),
                        os.path.join(campaign, "openedge"))
    with open(deck_path, "w") as f:
        f.write(deck)
    print(f"  rewrote deck paths -> {deck_path}")

    # 4. campaign config.json (loop_driver entry point)
    camp_cfg = dict(cfg)
    camp_cfg["coupled_dir"] = campaign
    camp_cfg["solps_base_dir"] = "solps/baserun"
    # scripts + binary are shared/read-only -> keep canonical absolute paths
    camp_cfg["solps_run_script"] = absp(cfg.get("solps_run_script", "run_solps.csh"))
    camp_cfg["wlld_script"] = absp(cfg.get("wlld_script", "run_b2plot_wlld.sh"))
    camp_cfg["plasma_h5"] = os.path.join(camp_oe, "plasma", f"plasma_{base}.h5")
    cases = dict(camp_cfg["cases"])
    cases[case] = dict(cases[case],
                       solps_run_dir=os.path.join("solps", case),
                       oe_deck=os.path.join("openedge", "full_runs", case, "in.oedge_wall"),
                       oe_output_dir=os.path.join("openedge", "full_runs", case, "output"))
    camp_cfg["cases"] = cases
    # promote the selected case's droplet_emit to top-level (oneshot_driver
    # reads cfg["droplet_emit"] -> -var ndrop for the deck's `n ${ndrop}`)
    if cases[case].get("droplet_emit") is not None:
        camp_cfg["droplet_emit"] = cases[case]["droplet_emit"]
    camp_cfg.setdefault("loop", {})
    camp_cfg["loop"] = dict(camp_cfg["loop"], case=case)
    # plasma_build: b2fgmtry/equ resolve under campaign via the baserun symlink;
    # wall_in stays absolute (canonical, read-only).
    pb = dict(camp_cfg.get("plasma_build", {}))
    if pb.get("wall_in"):
        pb["wall_in"] = absp(pb["wall_in"])
    camp_cfg["plasma_build"] = pb

    camp_cfg_path = os.path.join(campaign, "config.json")
    with open(camp_cfg_path, "w") as f:
        json.dump(camp_cfg, f, indent=4)

    print(f"\nstaged. run the loop with:")
    print(f"  python3 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loop_driver.py')} {camp_cfg_path}")
    print(f"campaign config: {camp_cfg_path}")


if __name__ == "__main__":
    main()

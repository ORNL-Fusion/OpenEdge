#!/usr/bin/env bash
set -euo pipefail

python3 read_tally_dump.py tmp.collision.wall.I.mode1.nemit200 --out-prefix tally.wall.I.mode1.nemit200
python3 read_tally_dump.py tmp.collision.wall.O.mode1.nemit200 --out-prefix tally.wall.O.mode1.nemit200

echo "Done. Generated:"
echo "  tally.wall.I.mode1.nemit200.steps.csv"
echo "  tally.wall.I.mode1.nemit200.records.csv"
echo "  tally.wall.O.mode1.nemit200.steps.csv"
echo "  tally.wall.O.mode1.nemit200.records.csv"

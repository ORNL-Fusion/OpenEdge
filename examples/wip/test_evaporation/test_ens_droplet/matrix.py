import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_surf_blocks(path):
    """
    Read ALL 'ITEM: SURFS' blocks from a SPARTA surf dump.
    Returns list of (timestep, DataFrame) tuples sorted by timestep.
    """
    blocks = []
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    current_ts = None
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            current_ts = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: SURFS"):
            col_names = line.split()[2:]  # e.g. ['id', 'f_f_inner[*]', 'f_f_outer[*]']
            i += 1
            rows = []
            while i < len(lines) and not lines[i].strip().startswith("ITEM:"):
                parts = lines[i].split()
                if parts:
                    row = {col: float(v) for col, v in zip(col_names, parts)}
                    rows.append(row)
                i += 1
            if rows:
                df = pd.DataFrame(rows)
                df['id'] = df['id'].astype(int)
                blocks.append((current_ts, df))
        else:
            i += 1
    return sorted(blocks, key=lambda x: x[0])


# ── load and extract total integer hit counts ─────────────────────────────
# fix ave/surf uses "ave running": value at timestep T = total_hits / T.
# → To recover total integer hits, use the last dump block and multiply by T.
blocks = read_surf_blocks("tmp.all.surf")
blocks = [(ts, df) for ts, df in blocks if ts > 0]   # skip step-0 init dump

if not blocks:
    raise RuntimeError("No data blocks found — run the simulation first.")

last_ts, last_df = blocks[-1]   # last block has the most accumulated statistics

sp_cols = [c for c in last_df.columns if c != 'id']

# running mean × timestep → total hit count; round to nearest integer
totals = last_df.set_index('id')[sp_cols].multiply(last_ts).round().astype(int)
totals.columns = ['inner', 'outer']   # sp_cols[0]=inner, sp_cols[1]=outer

# keep only surf elements that received at least one hit
hit_mask = totals.sum(axis=1) > 0
matrix = totals[hit_mask].T
matrix.index.name = 'source'
matrix.columns.name = 'surf_id'

# ── plot ──────────────────────────────────────────────────────────────────
n_surfs = matrix.shape[1]
plt.figure(figsize=(max(8, n_surfs * 0.6), 3), dpi=150)
sns.heatmap(matrix, annot=True, fmt="d", cmap="viridis",
            linewidths=0.5, cbar_kws={"label": "total particle hits"})
plt.xlabel("surface ID")
plt.ylabel("source species")
plt.title(f"Transfer matrix — total particle hits  (T = {last_ts} steps)")
plt.tight_layout()
plt.savefig("Figs/transfer_matrix.png", dpi=200)
plt.show()

#import h5py
#
#file_path = "plasma.h5"  # adjust if needed
#
#with h5py.File(file_path, "r") as f:
#    def visit(name, obj):
#        if isinstance(obj, h5py.Dataset):
#            print(f"{name} : shape={obj.shape}, dtype={obj.dtype}")
#        elif isinstance(obj, h5py.Group):
#            print(f"{name} (group)")
#    f.visititems(visit)
#
#
#print(structure)
#
#exit()


import h5py
import numpy as np

file_path = "/Users/42d/OpenEdge_Liquid_Metal/examples/test_droplet/NewSimulation/input/plasma.h5"  # adjust if needed
#file_path = "/Users/42d/OpenEdge_ORNL_GITHUB/examples/test_west/input/plasma.h5"  # adjust if needed

with h5py.File(file_path, "r") as f:
    print(f.keys())
#    r= f['r'][:]
#    z= f['z'][:]
#    te = f['temp_e'][:]
#    print(te)
#    
exit()
    

from matplotlib import pyplot as plt


plt.pcolormesh(r,z,te)
plt.show()

exit()
def list_datasets(f):
    paths = []
    def cb(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    f.visititems(cb)
    return paths

def find_candidates(f, aliases):
    aliases = [a.lower() for a in aliases]
    hits = []
    for name in list_datasets(f):
        low = name.lower()
        base = low.rsplit("/", 1)[-1]
        if any(a in base or a in low for a in aliases):
            hits.append(name)
    return sorted(set(hits))

with h5py.File(file_path, "r") as f:
    # Try common aliases
    te_hits = find_candidates(f, ["temp_e"])
    ti_hits = find_candidates(f, ["temp_i"])

    print("Found Te datasets:", te_hits or "None")
    print("Found Ti datasets:", ti_hits or "None")

    def show(path):
        d = f[path]
        # Load data (if huge, you can slice instead, e.g., d[..., -1])
        a = d[()]
        print(f"\n{path}: shape={a.shape}, dtype={a.dtype}")
        try:
            print(f"  stats: min={np.nanmin(a)}, max={np.nanmax(a)}, mean={np.nanmean(a)}")
        except Exception as e:
            print(f"  (could not compute stats: {e})")
        if len(d.attrs):
            print("  attrs:", {k: d.attrs[k] for k in d.attrs.keys()})
        return a

    te = show(te_hits[0]) if te_hits else None
    ti = show(ti_hits[0]) if ti_hits else None

# --- OPTIONAL: quick plots if you want a look ---
if False:
    import matplotlib.pyplot as plt

    def quick_plot(A, title):
        if A is None:
            return
        if A.ndim == 1:
            plt.figure(); plt.plot(A); plt.title(title); plt.xlabel("index"); plt.ylabel(title); plt.show()
        elif A.ndim == 2:
            plt.figure(); im = plt.imshow(A, origin="lower", aspect="auto"); plt.title(title); plt.colorbar(im); plt.show()
        elif A.ndim == 3:
            idx = -1  # last slice
            plt.figure(); im = plt.imshow(A[idx], origin="lower", aspect="auto")
            plt.title(f"{title} [slice {idx}]"); plt.colorbar(im); plt.show()

    quick_plot(te, "Te")
    quick_plot(ti, "Ti")

exit()
import h5py
import numpy as np

file_path = "plasma.h5"

with h5py.File(file_path, "r+") as f:
    if "parr_flow_r" not in f:
        raise KeyError("Dataset 'parr_flow_r' not found")

    src = f["parr_flow_r"]
    data = src[()]  # read into memory

    # If v_parr exists, delete it so we can recreate
    if "parr_flow" in f:
        del f["parr_flow"]

    # Create v_parr with the same shape/dtype as parr_flow_r
    dset = f.create_dataset(
        "parr_flow",
        data=data,
        dtype=src.dtype,        # keep same dtype (likely float64)
        # Optional: enable chunking/compression if the file uses it elsewhere
        # chunks=src.chunks,
        # compression=src.compression,
        # compression_opts=src.compression_opts,
    )

    # (Optional) copy over attributes from parr_flow_r
    for k, v in src.attrs.items():
        dset.attrs[k] = v

    # (Optional) add a brief description
    dset.attrs["description"] = "Copied from parr_flow_r"

# Quick verification (optional)
with h5py.File(file_path, "r") as f:
    print("v_parr :", f["parr_flow"].shape, f["parr_flow"].dtype)
    # Sanity check equality
    print("Equal to parr_flow_r?", np.array_equal(f["parr_flow"][()], f["parr_flow_r"][()]))

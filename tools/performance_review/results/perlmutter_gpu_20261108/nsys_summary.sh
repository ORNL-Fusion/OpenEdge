#!/bin/bash
# usage: nsys_summary.sh REPORT.nsys-rep NSTEPS  -> prints per-step kernel/memcpy/API totals and top entries
set -euo pipefail
rep=$1; nsteps=$2; NSYS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
out=${rep%.nsys-rep}_stats; mkdir -p $out
$NSYS stats --force-export=true -q --format csv -o $out/r --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_pushpop_sum,nvtx_kern_sum $rep >/dev/null
python3 - $out $nsteps <<'PY'
import csv,sys,glob,os
out,n=sys.argv[1],int(sys.argv[2])
def load(name):
    f=glob.glob(f"{out}/r_{name}.csv")
    if not f: return []
    return list(csv.DictReader(open(f[0])))
def ns(x): return float(x)/1e9
print(f"== per-step totals over {n} steps (profiled process only)")
k=load("cuda_gpu_kern_sum"); tk=sum(ns(r["Total Time (ns)"]) for r in k); nk=sum(int(r["Instances"]) for r in k)
print(f"GPU kernel time: {tk:.3f} s total, {1e3*tk/n:.3f} ms/step, {nk/n:.1f} launches/step, {len(k)} distinct kernels")
m=load("cuda_gpu_mem_time_sum"); ms=load("cuda_gpu_mem_size_sum")
for r in m: print(f"memop {r['Operation']:28s} {ns(r['Total Time (ns)']):8.3f} s  {int(r['Count'])/n:8.1f}/step")
for r in ms: print(f"bytes {r['Operation']:28s} {float(r['Total (MB)']):10.1f} MB total  {1e3*float(r['Total (MB)'])/n:10.1f} kB/step")
a=load("cuda_api_sum")
print("-- CUDA API (host) top by time")
for r in sorted(a,key=lambda r:-float(r["Total Time (ns)"]))[:12]:
    print(f"  {r['Name']:36s} {ns(r['Total Time (ns)']):8.3f} s {int(r['Num Calls'])/n:8.1f} calls/step")
print("-- top kernels by time")
for r in sorted(k,key=lambda r:-float(r["Total Time (ns)"]))[:25]:
    nm=r["Name"]; nm=nm[:150]
    print(f"  {ns(r['Total Time (ns)']):7.3f} s {int(r['Instances'])/n:7.2f}/step {float(r['Avg (ns)'])/1e3:9.1f} us  {nm}")
nv=load("nvtx_pushpop_sum")
print("-- top NVTX (Kokkos labels, host-side range time) by time")
for r in sorted(nv,key=lambda r:-float(r["Total Time (ns)"]))[:40]:
    print(f"  {ns(r['Total Time (ns)']):7.3f} s {int(r['Instances'])/n:7.2f}/step  {r['Range'][:140]}")
nk=load("nvtx_kern_sum")
if nk:
    print("-- kernel time attributed to NVTX range (nvtx_kern_sum)")
    agg={}
    for r in nk:
        key=r.get("NVTX Range") or r.get("Range")
        agg.setdefault(key,[0,0]); agg[key][0]+=float(r["Total Time (ns)"]); agg[key][1]+=int(r["Instances"]) if "Instances" in r else 0
    for key,(t,c) in sorted(agg.items(),key=lambda kv:-kv[1][0])[:30]:
        print(f"  {ns(t):7.3f} s {c/n:7.2f}/step  {key[:140]}")
PY

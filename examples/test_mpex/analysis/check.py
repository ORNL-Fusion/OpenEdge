import collections

def check_sparta_surface(fname):
    with open(fname) as f:
        txt = f.read()

    pts_block = txt.split("Points\n",1)[1].split("Triangles\n",1)[0]
    tri_block = txt.split("Triangles\n",1)[1]

    pts = {}
    for line in pts_block.splitlines():
        line = line.strip()
        if not line:
            continue
        pid,x,y,z = line.split()
        pts[int(pid)] = (float(x), float(y), float(z))

    tris = []
    for line in tri_block.splitlines():
        line = line.strip()
        if not line:
            continue
        tid,i,j,k = line.split()
        tris.append((int(i),int(j),int(k)))

    edge_counts = collections.Counter()
    for i,j,k in tris:
        for a,b in ((i,j),(j,k),(k,i)):
            key = tuple(sorted((a,b)))
            edge_counts[key] += 1

    counts = edge_counts.values()
    print("num edges:", len(edge_counts))
    print("min count:", min(counts), "max count:", max(counts))

    bad = [e for e,c in edge_counts.items() if c != 2]
    print("edges with count != 2:", len(bad))

check_sparta_surface("skimmer.txt")


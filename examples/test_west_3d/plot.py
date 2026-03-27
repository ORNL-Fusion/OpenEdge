import numpy as np, matplotlib.pyplot as plt

with open('input/data/west_3mw.equ') as f:
  text = f.read()

# Parse by markers
import re
jm = int(re.search(r'jm\s*=\s*(\d+)', text).group(1))
km = int(re.search(r'km\s*=\s*(\d+)', text).group(1))
psib = float(re.search(r'psib\s*=\s*([\d.eE+-]+)', text).group(1))

def read_after(marker, n):
  pos = text.find(marker)
  chunk = text[pos + len(marker):]
  vals = []
  for tok in chunk.split():
      if len(vals) >= n: break
      try: vals.append(float(tok))
      except:
          if len(vals) > 0: break
  return np.array(vals[:n])

r = read_after('r(1:jm);', jm)
z = read_after('z(1:km);', km)
psi_flat = read_after('((psi(j,k)-psib,j=1,jm),k=1,km)', jm*km)
psi = psi_flat.reshape(km, jm) + psib

psi_axis = psi.min()
psi_norm = (psi - psi_axis) / (psib - psi_axis)

pts = []
with open('input/data/wall.txt') as f:
  sec = None
  for ln in f:
      s = ln.strip()
      if s == 'Points': sec = 'p'; continue
      if s == 'Lines': break
      if sec == 'p' and s:
          c = s.split()
          if len(c) >= 3: pts.append((float(c[1]), float(c[2])))
pts = np.array(pts)

fig, ax = plt.subplots(figsize=(6,8))
R, Z = np.meshgrid(r, z)
cs = ax.contourf(R, Z, psi_norm, levels=np.linspace(0, 1.5, 30), cmap='RdYlBu_r')
ax.contour(R, Z, psi_norm, levels=[0.9, 0.95, 1.0, 1.05], colors='k', linewidths=1.5)
ax.plot(pts[:,0], pts[:,1], 'k-', lw=2)
plt.colorbar(cs, label='psi_norm')
ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
ax.set_title('WEST 3MW psi_norm')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('output/psi_norm_west.png', dpi=150)
plt.show()
print('Saved output/psi_norm_west.png')
print(f'psi_axis={psi_axis:.6f}, psib={psib:.6f}')

#!/usr/bin/env python3
"""Draw the auxiliary-region architecture; this is a schematic, not simulation data."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openedge-cartoon-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import numpy as np

out = Path(__file__).resolve().parent / 'figures'
out.mkdir(exist_ok=True)
blue, orange, green, ink = '#2468A0', '#C96720', '#227A67', '#223146'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(14, 8.6), facecolor='white')
fig.text(.055,.95,'Two auxiliary bands connect the core density to Monte Carlo particles',
         size=20, weight='bold', color=ink)
fig.text(.055,.91,'Hybrid particle–continuum coupling  •  diffusion-only harness  •  schematic, not to scale',
         size=12, color='#596779')

ax = fig.add_axes([.055,.46,.89,.40])
ax.set_xlim(-4.9,4.9); ax.set_ylim(-1.05,3.0); ax.axis('off')
ax.add_patch(Rectangle((-4.7,0),4.7,1.55,facecolor='#EFF5FB',edgecolor='#CEDAE6'))
ax.add_patch(Rectangle((0,0),4.7,1.55,facecolor='#FFF5EB',edgecolor='#E8D5C5'))
ax.add_patch(Rectangle((-.95,0),.95,1.55,facecolor='#C1D9F0',edgecolor=blue,lw=1.5))
ax.add_patch(Rectangle((0,0),.95,1.55,facecolor='#F6CFAB',edgecolor=orange,lw=1.5))
# The blue bars represent finite-volume density values. Particle heights are illustrative.
for j, x in enumerate(np.arange(-4.6,-.15,.445)):
    height = .59+.13*np.sin(j*.65)
    ax.add_patch(Rectangle((x,.12),.38,height,facecolor=blue,alpha=.68,edgecolor='white',lw=.6))
rng=np.random.default_rng(42)
x=rng.uniform(.12,4.55,67); y=rng.uniform(.18,1.32,len(x))
ax.scatter(x,y,s=24,color=orange,alpha=.86,zorder=3)
ax.plot([0,0],[-.10,1.70],color=ink,lw=1.8,ls=(0,(4,3)),zorder=5)
ax.text(-2.85,1.78,'1-D CORE: density / PDE',ha='center',color=blue,size=15,weight='bold')
ax.text(2.85,1.78,'MONTE CARLO: particles',ha='center',color=orange,size=15,weight='bold')
ax.text(-2.8,1.18,'PDE evolves here',ha='center',color=blue,size=13,
        bbox=dict(facecolor='#EFF5FB',edgecolor='none',pad=3))
ax.text(2.9,.76,'Particles move here',ha='center',color=orange,size=13,
        bbox=dict(facecolor='#FFF5EB',edgecolor='none',alpha=.94,pad=3))
for y, start, end in [(2.46,-.75,.75),(2.02,.75,-.75)]:
    ax.add_patch(FancyArrowPatch((start,y),(end,y),arrowstyle='-|>',mutation_scale=20,lw=2.5,color=green))
ax.text(0,2.70,'core → particles: convert density to marker weight',ha='center',color=green,size=12)
ax.text(0,2.18,'particles → core: convert marker weight to density',ha='center',color=green,size=12)
for lo,hi,label,col in [(-.95,0,'inner auxiliary band',blue),(0,.95,'outer auxiliary band',orange)]:
    ax.annotate('',xy=(lo,-.18),xytext=(hi,-.18),arrowprops=dict(arrowstyle='|-|',color=col,lw=1.2))
    ax.text((lo+hi)/2,-.42,label.replace(' auxiliary','\nauxiliary'),ha='center',va='top',color=col,size=11)
ax.text(0,-.97,'interface / core cut',ha='center',color=ink,size=12,weight='bold')
ax.annotate('toward the axis',xy=(-4.6,-.63),xytext=(-2.7,-.63),va='center',ha='center',color='#687486',
            arrowprops=dict(arrowstyle='->',color='#687486'))
ax.annotate('toward the edge',xy=(4.6,-.63),xytext=(2.7,-.63),va='center',ha='center',color='#687486',
            arrowprops=dict(arrowstyle='->',color='#687486'))

# Three explanatory cards. Each band keeps its native model; only exchange is shared.
cards=[(.055,'INNER BAND',blue,'Last few core cells\nInventory comes from density × volume.'),
       (.36,'OUTER BAND',orange,'Nearby Monte Carlo markers\nInventory comes from summed weights.'),
       (.665,'EXCHANGE',green,'Transfer in both directions\nEqual density gives zero mean net flux.')]
for left,title,col,body in cards:
    box=FancyBboxPatch((left,.215),.28,.17,boxstyle='round,pad=.012,rounding_size=.012',
                      transform=fig.transFigure,facecolor='#F7F9FC',edgecolor='#DDE3EB',lw=1)
    fig.add_artist(box)
    fig.text(left+.012,.349,title,color=col,size=12,weight='bold')
    fig.text(left+.012,.309,body,color=ink,size=11.5,linespacing=1.7,va='top')
fig.text(.055,.157,'Each band is evolved by its own model. The shared exchange step transfers inventory between them.',
         color=ink,size=13,weight='bold')
fig.text(.055,.113,'These are adjacent auxiliary regions. An overlapping buffer would instead extend both solvers into the same space.',
         color='#596779',size=11.5)
fig.text(.055,.073,'Band widths are numerical resolution settings; they must be refined together with the exchange interval.',
         color='#596779',size=11.5)
fig.text(.055,.025,'Architecture: auxiliary region method (Smith & Yates, 2018, doi:10.1098/rsos.180920). '
         'Our harness uses an interval-based two-box exchange variant.',color='#798594',size=9)
for ext in ('png','svg','pdf'):
    fig.savefig(out/f'auxiliary_region_cartoon.{ext}',dpi=190,facecolor='white')
plt.close(fig)
print(out/'auxiliary_region_cartoon.png')

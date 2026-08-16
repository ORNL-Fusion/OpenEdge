#!/usr/bin/env python3
"""Create the always-present, executable RFPIE case inspection notebook."""

from pathlib import Path

import nbformat as nbf


CASE_DIR = Path(__file__).resolve().parents[1]


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    markdown("""
# RFPIE He-on-W OpenEdge case audit

Run this notebook before transport. It checks what OpenEdge will actually see:
the probe-derived He plasma, SI geometry, per-tile DC/RF waveform, RustBCA
He-on-W data, and the resulting sputtered-W source. Legacy surface data from the
original case are intentionally not used.
"""),
    code("""
from pathlib import Path
import glob, json, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

case = Path.cwd().resolve()
if case.name == 'scripts':
    case = case.parent
if not (case / 'input/config.json').exists():
    case = Path('examples/workflows/impurity_transport/rfpie_tungsten_transport').resolve()
process_path = next(p / 'database/processes.h5' for p in case.parents
                    if (p / 'database/processes.h5').exists())
assert process_path.exists(), f'Missing OpenEdge process database: {process_path}'
cfg = json.loads((case / 'input/config.json').read_text())
summary = json.loads((case / 'input/geometry_summary.json').read_text())

def read_dump(path):
    lines = Path(path).read_text().splitlines()
    step = int(lines[1]); count = int(lines[3])
    columns = lines[8].split()[2:]
    if count:
        values = np.asarray([[float(v) for v in row.split()]
                             for row in lines[9:9+count]])
    else:
        values = np.empty((0, len(columns)))
    return step, {name: values[:, i] for i, name in enumerate(columns)}

print('case:', case)
print(json.dumps(summary, indent=2))
"""),
    markdown("""
## 1. Plasma data

The source Excel workbook labels its horizontal axis **Radius (cm)**. The old
helper script silently multiplied those positions by `1e-3`, treating ±5 cm
as ±5 mm. That factor-of-ten error created the visible shoulder at 5 mm. The
actual measurement reaches 50 mm, so this 31 mm OpenEdge domain—and the entire
11.43 mm-radius target—lies within measured data. The dashed curves reproduce
the incorrect-unit reconstruction from the screenshot.

The two scans are averaged and folded about the axis, with their reported
errors and left/right asymmetry retained. Density uses shape-preserving
interpolation of `log(ne-floor)`. Electron temperature is fit with an
uncertainty-weighted polynomial in `r^2`, which is smooth on axis. A
slope-matched exponential continuation is defined beyond 50 mm but is not used
inside this case.

The axial profile remains uniform because the scan has no axial information.
Caughman et al. (IEEE TPS 52, 2024, DOI 10.1109/TPS.2024.3374252) supports a
4–6 eV edge rise qualitatively, but its plotted profile is a different plasma
condition and is not substituted for these measurements.
"""),
    code("""
with h5py.File(case / 'input/plasma_he.h5', 'r') as h5:
    r = h5['r'][:]; z = h5['z'][:]
    ne = h5['dens_e'][:]; te = h5['temp_e'][:]; ti = h5['temp_i'][:]
    rs = h5['audit/lp_signed_r_m'][:]
    ne_raw = h5['audit/lp_mean_ne_m3'][:]
    ne_raw_sigma = h5['audit/lp_mean_ne_sigma_m3'][:]
    te_raw = h5['audit/lp_mean_te_eV'][:]
    te_raw_sigma = h5['audit/lp_mean_te_sigma_eV'][:]
    rf = h5['audit/folded_r_m'][:]
    nef = h5['audit/folded_ne_m3'][:]
    nef_sigma = h5['audit/folded_ne_sigma_m3'][:]
    tef = h5['audit/folded_te_eV'][:]
    tef_sigma = h5['audit/folded_te_sigma_eV'][:]
    ne_legacy = h5['audit/legacy_wrong_unit_ne_r_m3'][:]
    te_legacy = h5['audit/legacy_wrong_unit_te_r_eV'][:]
    ne_tail_scale = h5['audit/density_tail_scale_m'][()]
    te_tail_scale = h5['audit/temperature_tail_scale_m'][()]
    te_edge = h5['audit/temperature_edge_eV'][()]
    te_edge_slope = h5['audit/temperature_edge_slope_eV_per_m'][()]
    plasma_attrs = dict(h5.attrs)
    rw = h5['wall_flux/r'][:]
    gamma_he = h5['wall_flux/gamma_i'][0]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(rs*1e3, ne_raw, yerr=ne_raw_sigma, fmt='o', alpha=.45,
               ms=4, label='scan mean, signed')
ax[0].errorbar(rf*1e3, nef, yerr=nef_sigma, fmt='ko', ms=4, label='folded')
ax[0].plot(r*1e3, ne_legacy, '--', color='0.55', label='old wrong-unit reconstruction')
ax[0].plot(r*1e3, ne[0], lw=2, label='smooth OpenEdge profile')
ax[0].set(xlabel='radius [mm]', ylabel=r'$n_e$ [m$^{-3}$]', yscale='log')
ax[1].errorbar(rs*1e3, te_raw, yerr=te_raw_sigma, fmt='o', alpha=.45,
               ms=4, label='scan mean, signed')
ax[1].errorbar(rf*1e3, tef, yerr=tef_sigma, fmt='ko', ms=4, label='folded')
ax[1].plot(r*1e3, te_legacy, '--', color='0.55', label='old wrong-unit reconstruction')
ax[1].plot(r*1e3, te[0], lw=2, label='smooth OpenEdge profile')
ax[1].set(xlabel='radius [mm]', ylabel=r'$T_e$ [eV]')
for a in ax:
    a.axvline(rf[-1]*1e3, color='k', lw=.8, ls=':', label='last measured radius')
    a.axvline(rw[-1]*1e3, color='tab:red', lw=.8, ls=':', label='target edge')
    a.grid(alpha=.25); a.legend(fontsize=8)
plt.tight_layout()
print(f'grid={ne.shape}, ne={ne.min():.3e}..{ne.max():.3e} m^-3, '
      f'Te={te.min():.2f}..{te.max():.2f} eV, Ti={ti.min():.3f} eV')
print('position-unit evidence:', plasma_attrs['source_unit_evidence'])
print(f'measured edge = {rf[-1]*1e3:.1f} mm; density tail e-folding length = {ne_tail_scale*1e3:.3f} mm')
print(f'Te fit at measured edge = {te_edge:.3f} eV, slope = {te_edge_slope*1e-3:.3f} eV/mm, '
      f'tail length = {te_tail_scale*1e3:.3f} mm')
print('density model:', plasma_attrs['density_model'])
print('temperature model:', plasma_attrs['temperature_model'])
print('axial assumption:', plasma_attrs['assumption_axial_profile'])
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
im0 = ax[0].pcolormesh(r*1e3, z*1e3, ne, shading='auto',
                       norm=LogNorm(vmin=ne.min(), vmax=ne.max()))
im1 = ax[1].pcolormesh(r*1e3, z*1e3, te, shading='auto')
fig.colorbar(im0, ax=ax[0], label=r'$n_e$ [m$^{-3}$]')
fig.colorbar(im1, ax=ax[1], label=r'$T_e$ [eV]')
ax[0].set_title('He plasma density'); ax[1].set_title('electron temperature')
for a in ax:
    a.axvline(rw[-1]*1e3, color='r', lw=.7, ls=':')
    a.set(xlabel='R [mm]', ylabel='z [mm]', xlim=(0,r[-1]*1e3))
"""),
    markdown("""
## 2. Geometry and target tiles

The original binary STL coordinates are millimetres. The generated SPARTA
surfaces below are metres. Only the planar +z face is in the sputtering and
sheath group; target sides/back are absorbing but unbiased.
"""),
    code("""
def read_surf(path):
    lines = path.read_text().splitlines()
    npnt = int(next(x.split()[0] for x in lines if x.strip().endswith('points')))
    ntri = int(next(x.split()[0] for x in lines if x.strip().endswith('triangles')))
    ip = lines.index('Points') + 2
    pts = np.array([[float(v) for v in lines[ip+i].split()[1:4]] for i in range(npnt)])
    it = lines.index('Triangles') + 2
    rows = [lines[it+i].split() for i in range(ntri)]
    tri = np.array([[int(v)-1 for v in row[1:4]] for row in rows])
    custom = np.array([[float(v) for v in row[4:]] for row in rows]) if len(rows[0]) > 4 else None
    return pts, tri, custom

pd, td, _ = read_surf(case/'input/domain.surf')
pb, tb, _ = read_surf(case/'input/target_body.surf')
pf, tf, sheath = read_surf(case/'input/target_face.surf')
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121, projection='3d')
ax.add_collection3d(Poly3DCollection(pd[td], facecolor='tab:blue', alpha=.035, edgecolor='tab:blue', linewidth=.12))
ax.add_collection3d(Poly3DCollection(pb[tb], facecolor='0.45', alpha=.45, linewidth=.1))
ax.add_collection3d(Poly3DCollection(pf[tf], facecolor='tab:red', alpha=.85, linewidth=.15))
ax.set(xlim=(-.031,.031), ylim=(-.031,.031), zlim=(-.001,.061), xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title='full chamber and W target')
ax2 = fig.add_subplot(122, projection='3d')
coll = Poly3DCollection(pf[tf], array=sheath[:,0], cmap='coolwarm', edgecolor='k', linewidth=.2)
ax2.add_collection3d(coll); fig.colorbar(coll, ax=ax2, shrink=.65, label='Vdc [V]')
ax2.set(xlim=(-.012,.012), ylim=(-.012,.012), zlim=(.0018,.0020), xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=f'{len(tf)} biased face tiles')
plt.tight_layout()
print('unique [Vdc, Vrf, phase] rows:', np.unique(sheath, axis=0))
"""),
    markdown("""
## 3. Tile voltage and sheath convention

`Vwall(t) = Vdc + Vrf_peak sin(2 pi f t + phase)`, relative to the local
quasineutral plasma. A negative wall adds `max(0,-Vwall)` eV per unit charge
to the incident He energy. The normal floating-sheath drop is retained. The
transport mesh does not resolve the Debye sheath; `sheath boundary` is the
appropriate sub-grid potential jump.

For nonzero RF voltage, the present source averages the nonlinear sputter
yield over instantaneous sinusoidal voltage samples. It does **not** reproduce
the bimodal kinetic ion-energy distribution calculated with hPIC2 in the
paper. DC is therefore the quantitatively cleaner first case; RF remains a
documented reduced model until an IEDF table or resolved sheath is supplied.
"""),
    code("""
vdc, vrf, phase = sheath[0]
freq = cfg['target_sheath']['frequency_hz']
t = np.linspace(0, 2/freq, 500)
vwall = vdc + vrf*np.sin(2*np.pi*freq*t + phase)
plt.figure(figsize=(8,3.5)); plt.plot(t*1e9, vwall)
plt.axhline(0, color='k', lw=.6); plt.grid(alpha=.25)
plt.xlabel('time [ns]'); plt.ylabel(r'$V_{wall}$ [V]')
plt.title(f'{freq/1e6:.2f} MHz target waveform')
period = 1/freq; dt = cfg['transport']['timestep_s']
print(f'RF period={period*1e9:.3f} ns; dt={dt*1e9:.3f} ns; steps/period={period/dt:.1f}')
EPS0=8.8541878128e-12; QE=1.602176634e-19
mi=4.002602*1.66053906660e-27
lambda_D=np.sqrt(EPS0*te[0,0]/(ne[0,0]*QE))
cs0=np.sqrt((te[0,0]+ti[0,0])*QE/mi)
fpi=np.sqrt(ne[0,0]*QE**2/(EPS0*mi))/(2*np.pi)
grid_cells=np.asarray(cfg['transport']['grid_cells'])
box_lengths=np.array([.0602,.0602,.0602])
cell_widths=box_lengths/grid_cells
dx_min=cell_widths.min()
vmax_w=np.sqrt(2*80*QE/(184*1.66053906660e-27))
step_fraction=vmax_w*dt/dx_min
substep_fraction=step_fraction/cfg['transport']['pusher_subcycles']
diffusion_step=np.sqrt(2*cfg['transport']['d_perp_m2_s']*dt)
print(f'grid={tuple(grid_cells)}; cell widths [mm]={cell_widths*1e3}')
print(f'center Debye length={lambda_D*1e6:.1f} um; smallest cell={dx_min*1e3:.3f} mm '
      f'({dx_min/lambda_D:.0f} Debye lengths)')
print(f'80 eV W full-step/smallest-cell={step_fraction:.3f}; '
      f'Boris-substep/cell={substep_fraction:.3f}; '
      f'cross-field RMS step={diffusion_step*1e6:.1f} um')
print(f'center He Bohm speed={cs0:.3e} m/s; ion plasma frequency={fpi/1e6:.2f} MHz')
if vrf != 0 and period/dt < 20:
    print('WARNING: fewer than 20 steps/RF period; reduce dt for time-resolved RF transport')
"""),
    markdown("""
## 4. RustBCA He-on-W yield and implied W source

The notebook and OpenEdge both read the installed RustBCA table at
`/surface/sputter/he_on_w/{E,theta,Y}` in `database/processes.h5`.
"""),
    code("""
with h5py.File(process_path, 'r') as h5:
    table = h5['surface/sputter/he_on_w']
    Eaxis = table['E'][:]; Aaxis = table['theta'][:]; Y = table['Y'][:]
    table_source = table.attrs.get('source', '')
print(table_source, 'shape=', Y.shape, 'Y range=', (Y.min(),Y.max()))

fig, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
im=ax[0].pcolormesh(Aaxis,Eaxis,Y,shading='auto')
ax[0].set(xlabel='angle from normal [deg]',ylabel='He energy [eV]',yscale='log',title='RustBCA He on W'); fig.colorbar(im,ax=ax[0],label='W/He')
for a in [0,30,60,80]:
    ia=np.argmin(abs(Aaxis-a)); ax[1].plot(Eaxis,Y[:,ia],label=f'{Aaxis[ia]:.0f} deg')
ax[1].set(xlabel='He energy [eV]',ylabel='yield [W/He]',xscale='log',yscale='log',title='energy cuts')
ax[1].grid(alpha=.25); ax[1].legend()
"""),
    code("""
from scipy.interpolate import RegularGridInterpolator
AMU=1.66053906660e-27; ME=9.1093837015e-31
he_amu=4.002602
interpY=RegularGridInterpolator((np.log(Eaxis),Aaxis),Y,bounds_error=False,fill_value=None)

tew=np.interp(rw,r,te[0]); tiw=np.interp(rw,r,ti[0])
psi_over_te=0.5*np.log((he_amu*AMU)/(2*np.pi*ME*(1+tiw/tew)))
nphase=cfg['target_sheath']['phase_samples']
ph=np.linspace(0,2*np.pi,nphase,endpoint=False)
vw=vdc+vrf*np.sin(ph+phase)
Eimpact=2*tiw[:,None]+psi_over_te[:,None]*tew[:,None]+np.maximum(0,-vw)[None,:]
theta=np.zeros_like(Eimpact)  # configured Bz is normal to the planar target
Yphase=interpY(np.column_stack([np.log(np.clip(Eimpact.ravel(),Eaxis[0],Eaxis[-1])),theta.ravel()])).reshape(Eimpact.shape)
ybar=Yphase.mean(axis=1)
gamma_w=gamma_he*ybar

fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
ax[0].plot(ph, Eimpact[len(rw)//2]); ax[0].set(xlabel='RF phase [rad]',ylabel='He impact energy [eV]')
ax[1].plot(rw*1e3,gamma_he); ax[1].set(xlabel='target radius [mm]',ylabel=r'$\\Gamma_{He}$ [m$^{-2}$ s$^{-1}$]',yscale='log')
ax[2].plot(rw*1e3,gamma_w); ax[2].set(xlabel='target radius [mm]',ylabel=r'$\\Gamma_W$ [m$^{-2}$ s$^{-1}$]',yscale='log')
for a in ax: a.grid(alpha=.25)

rate=2*np.pi*np.trapezoid(gamma_w*rw,rw)
print(f'center: Te={tew[0]:.3f} eV, floating drop={psi_over_te[0]*tew[0]:.3f} V')
print(f'center He impact energy: {Eimpact[0].min():.2f}..{Eimpact[0].max():.2f} eV')
print(f'center phase-averaged yield: {ybar[0]:.4g} W/He')
print(f'integrated sputtered-W source: {rate:.4e} atoms/s')
"""),
    markdown("""
## 5. ADAS charge-state timescales

The local ionization time is `1 / (ne * SCD_q)`. Comparing it with the run
duration and a simple W flight time explains which charge states can develop.
The coefficients below are exactly the OpenADAS tables used by OpenEdge.
"""),
    code("""
with h5py.File(process_path,'r') as h5:
    scd=h5['volume/rates/scd/w/coefficient'][:]
    scd_te=h5['volume/rates/scd/w/temperature'][:]
    scd_ne=h5['volume/rates/scd/w/density'][:]

log_points=np.column_stack([np.log10(te[0]),np.log10(ne[0]/1e6)])
tau_ion=[]
for charge in range(4):
    log_rate=RegularGridInterpolator((scd_te,scd_ne),scd[charge],
                                     bounds_error=False,fill_value=None)(log_points)
    rate_coefficient_cm3s=10**log_rate
    tau_ion.append(1/(rate_coefficient_cm3s*(ne[0]/1e6)))
tau_ion=np.asarray(tau_ion)

E_th=np.linspace(1e-6,80,2000)
th_pdf=E_th/(E_th+8.68)**3
mean_E=np.trapezoid(E_th*th_pdf,E_th)/np.trapezoid(th_pdf,E_th)
mean_speed=np.sqrt(2*mean_E*QE/(184*AMU))
flight_time=(cfg['plasma']['z_max_m']-summary['target_z_m'])/mean_speed
run_time=cfg['transport']['run_steps']*cfg['transport']['timestep_s']

fig,ax=plt.subplots(figsize=(7.5,4.3),constrained_layout=True)
for charge,tau_q in enumerate(tau_ion):
    ax.semilogy(r*1e3,tau_q*1e6,label=f'W$^{{{charge}+}}$ to W$^{{{charge+1}+}}$')
ax.axhline(run_time*1e6,color='k',ls='--',label=f'nominal run ({run_time*1e6:.0f} us)')
ax.axhline(flight_time*1e6,color='0.4',ls=':',label=f'ballistic flight scale ({flight_time*1e6:.1f} us)')
ax.axvline(rf[-1]*1e3,color='tab:blue',ls=':',lw=.8)
ax.axvline(rw[-1]*1e3,color='tab:red',ls=':',lw=.8)
ax.set(xlabel='radius [mm]',ylabel='local ionization time [us]',
       title='ADAS W ionization times in prescribed plasma',ylim=(1,1e9))
ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
for charge,tau_q in enumerate(tau_ion):
    print(f'axis W{charge}+ -> W{charge+1}+ ionization time = {tau_q[0]*1e6:.2f} us')
print(f'Thompson mean energy={mean_E:.2f} eV; straight-line domain flight scale={flight_time*1e6:.2f} us')
"""),
    markdown("""
## 6. Gross erosion areal density

This is the gross sputtered-W fluence implied by the prescribed background
over the nominal run time, not net surface evolution. Dividing by bulk W
atomic density gives an equivalent removed depth. Redeposition is presently
absorbing and is not yet accumulated in a surface composition ledger.
"""),
    code("""
W_BULK_DENSITY=6.338e28  # atoms/m3
nominal_time=cfg['transport']['run_steps']*cfg['transport']['timestep_s']
eroded_areal=gamma_w*nominal_time
equiv_depth_nm=eroded_areal/W_BULK_DENSITY*1e9

fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
ax[0].plot(rw*1e3,eroded_areal)
ax[0].set(xlabel='target radius [mm]',ylabel=r'gross removed W [m$^{-2}$]',
          title=f'fluence over {nominal_time*1e6:.1f} us')
ax[1].plot(rw*1e3,equiv_depth_nm)
ax[1].set(xlabel='target radius [mm]',ylabel='equivalent W depth [nm]',
          title='gross erosion equivalent')
for a in ax: a.grid(alpha=.25)
print(f'nominal emitted W atoms = {rate*nominal_time:.4e}')
print(f'center gross fluence = {eroded_areal[0]:.4e} m^-2; '
      f'equivalent depth = {equiv_depth_nm[0]:.4e} nm')

# Once OpenEdge has initialized cpmi, compare its area-by-triangle integral
# against the fine radial quadrature above. This exposes under-resolved tiles.
target_dump=case/'output/target.0.dump'
if target_dump.exists():
    _,surf=read_dump(target_dump)
    order=np.argsort(surf['id'])
    face_area=.5*np.linalg.norm(np.cross(pf[tf][:,1]-pf[tf][:,0],
                                        pf[tf][:,2]-pf[tf][:,0]),axis=1)
    oe_rate=np.sum(face_area*surf['c_cpmi'][order])
    print(f'OpenEdge triangle-integrated source = {oe_rate:.4e} atoms/s; '
          f'difference from radial audit = {(oe_rate/rate-1)*100:+.3f}%')
else:
    print('No target.0.dump yet; triangle-integrated source check skipped.')
"""),
    markdown("""
## 7. Transport diagnostics (after a run)

The input deck now interval-averages pweight-aware densities separately for
W, W+, W2+, W3+, and W4+. The plots below follow the useful MPEX/slag pattern:
an R-z total-W map, charge-state profiles along the device axis (averaged in a
small central cylinder for statistics), and an in-flight W column/areal-density
map, `integral n_W dz`. These are transport diagnostics; the last panel is not
deposited wall inventory.

The second figure is a clean charge-resolved density profile over the
experimental 0–30 mm distance normal to the target. It reads the OpenEdge dump
directly and shows only the simulated species densities—no synthetic brightness
or line-radiation assumptions and no separate handoff format.
"""),
    markdown("""
### 7.1 Load the OpenEdge charge-state density dump

This cell explicitly finds the newest `output/rfpie_w_density.*.dump`, reads
the five `f_fWdens[*]` fields written by `in.openedge`, and prints the selected
file and column names before doing any plotting.
"""),
    code("""
def step_from_name(path):
    match=re.search(r'\\.(\\d+)\\.dump$',str(path))
    return int(match.group(1)) if match else -1

density_pattern=case/'output/rfpie_w_density.*.dump'
density_files=sorted((case/'output').glob('rfpie_w_density.*.dump'),
                     key=step_from_name)
G=None
dump_dt=cfg['transport']['timestep_s']
dump_grid=np.asarray(cfg['transport']['grid_cells'],dtype=int)
settings_path=case/'output/run_settings.txt'
if settings_path.exists():
    dump_settings=dict(token.split('=',1) for token in settings_path.read_text().split())
    dump_dt=float(dump_settings['dt_s'])
    dump_grid=np.array([int(dump_settings[k]) for k in ('nx','ny','nz')])
else:
    dump_settings={}
if not density_files:
    print('No files match:',density_pattern)
    print('Run at least diagnostic_every_steps, then rerun this cell.')
else:
    density_path=density_files[-1]
    step,G=read_dump(density_path)
    labels=['W$^0$','W$^+$','W$^{2+}$','W$^{3+}$','W$^{4+}$']
    fields=[f'f_fWdens[{i}]' for i in range(1,6)]
    dens_species=np.vstack([G[name] for name in fields])
    dens_total=dens_species.sum(axis=0)
    xg,yg,zg,vol=G['xc'],G['yc'],G['zc'],G['vol']
    rg=np.hypot(xg,yg)
    print('loaded:',density_path)
    print('step:',step,'density fields:',fields)
    print('dump settings:',dump_settings if dump_settings else 'not recorded')
    print('dump time:',step*dump_dt*1e6,'us; dump grid:',tuple(dump_grid))
    configured_grid=np.asarray(cfg['transport']['grid_cells'])
    if dump_dt != cfg['transport']['timestep_s'] or np.any(dump_grid != configured_grid):
        print('NOTE: this is an earlier smoke result; current production defaults are',
              f'dt={cfg["transport"]["timestep_s"]*1e9:g} ns, grid={tuple(configured_grid)}')
"""),
    markdown("""
### 7.2 Plot the densities from that dump
"""),
    code("""
if G is None:
    print('No charge-state density data to plot.')
else:

    nx,ny,nz=dump_grid
    r_edges=np.linspace(0,.0301,nx//2+1)
    z_edges=np.linspace(-.0002,.0600,nz+1)
    atoms_rz=np.histogram2d(rg,zg,bins=[r_edges,z_edges],
                            weights=dens_total*vol)[0]
    volume_rz=np.histogram2d(rg,zg,bins=[r_edges,z_edges],weights=vol)[0]
    dens_rz=np.divide(atoms_rz,volume_rz,out=np.zeros_like(atoms_rz),
                      where=volume_rz>0)

    axis_radius=cfg['transport']['axis_average_radius_m']
    core=rg<=axis_radius
    core_volume=np.histogram(zg,bins=z_edges,weights=vol*core)[0]
    axial=[]
    for ds in dens_species:
        atom_hist=np.histogram(zg,bins=z_edges,weights=ds*vol*core)[0]
        axial.append(np.divide(atom_hist,core_volume,out=np.zeros_like(atom_hist),
                               where=core_volume>0))
    axial=np.asarray(axial)
    zmid=.5*(z_edges[:-1]+z_edges[1:])
    target_z=summary['target_z_m']
    distance_mm=(zmid-target_z)*1e3

    xy_edges=np.linspace(-.0301,.0301,nx+1)
    column_atoms=np.histogram2d(xg,yg,bins=[xy_edges,xy_edges],
                                weights=dens_total*vol)[0]
    dxy=xy_edges[1]-xy_edges[0]
    column_density=column_atoms/dxy**2

    fig,ax=plt.subplots(1,3,figsize=(16,4.3),constrained_layout=True)
    positive=dens_rz[dens_rz>0]
    if positive.size:
        norm=LogNorm(vmin=max(positive.min(),positive.max()*1e-5),vmax=positive.max())
        im=ax[0].pcolormesh(r_edges*1e3,z_edges*1e3,dens_rz.T,
                            shading='auto',norm=norm,cmap='magma')
        fig.colorbar(im,ax=ax[0],label=r'$n_W$ [m$^{-3}$]')
    ax[0].axhline(1.905,color='cyan',ls=':',lw=1)
    ax[0].set(xlabel='R [mm]',ylabel='z [mm]',title='total in-flight W density')

    for profile,label in zip(axial,labels):
        if np.any(profile>0): ax[1].semilogy(distance_mm,np.where(profile>0,profile,np.nan),label=label)
    ax[1].axvline(0,color='k',ls=':',lw=1,label='target face')
    ax[1].set(xlabel='distance from target [mm]',ylabel=r'$\\langle n_{W^q}\\rangle_{R<3mm}$ [m$^{-3}$]',
              title=f'charge states along axis (R < {axis_radius*1e3:.1f} mm)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)

    positive=column_density[column_density>0]
    if positive.size:
        norm=LogNorm(vmin=max(positive.min(),positive.max()*1e-5),vmax=positive.max())
        im=ax[2].pcolormesh(xy_edges*1e3,xy_edges*1e3,column_density.T,
                            shading='auto',norm=norm,cmap='viridis')
        fig.colorbar(im,ax=ax[2],label=r'$\\int n_W dz$ [m$^{-2}$]')
    ax[2].set(xlabel='x [mm]',ylabel='y [mm]',title='in-flight W column density',aspect='equal')

    inventory=np.sum(dens_species*vol,axis=1)
    total_inventory=inventory.sum()
    print(f'density frame step={step}, t={step*dump_dt*1e6:.3f} us')
    for label,value in zip(labels,inventory):
        print(f'  {label}: {value:.4e} atoms in domain '
              f'({value/total_inventory*100 if total_inventory else 0:.3f}%)')
    target_area=np.pi*rw[-1]**2
    print(f'total in-flight inventory / target area = '
          f'{total_inventory/target_area:.4e} m^-2')

    # Charge-resolved view over the measured spectroscopy distance.  The
    # OpenEdge grid dump above remains the authoritative data product.
    exp_range=(distance_mm>=0)&(distance_mm<=30)
    fig2,ax2=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    for profile,label in zip(axial,labels):
        ax2.semilogy(distance_mm[exp_range],
                     np.where(profile[exp_range]>0,profile[exp_range],np.nan),
                     '-o',ms=3,label=label)
    ax2.set(xlabel='distance normal to target [mm]',
            ylabel=r'$\\langle n_{W^q}\\rangle_{R<3mm}$ [m$^{-3}$]',
            title='OpenEdge W charge-state densities',xlim=(0,30))
    ax2.grid(alpha=.25); ax2.legend(fontsize=8,ncol=2)

    print('authoritative charge-state data:',density_files[-1])
    print(f'profile frame age={step*dump_dt*1e6:.2f} us; '
          f'run >= {flight_time*1e6:.1f} us before interpreting the full axial shape')

    particle_files=sorted((case/'output').glob('rfpie_w.*.dump'),key=step_from_name)
    if particle_files:
        pstep,P=read_dump(particle_files[-1])
        weighted=np.array([P['p_pweight'][P['type']==i].sum() for i in range(1,6)])
        print(f'latest particle frame step={pstep}; markers={len(P["type"])}; '
              f'pweight charge fractions={weighted/weighted.sum() if weighted.sum() else weighted}')
"""),
    markdown("""
## 8. Pre-run checklist

- Confirm the LP profiles and the unmeasured axial-profile assumption.
- Confirm target diameter/height and that only the red top tiles are biased.
- Confirm DC/RF voltage sign, RF frequency, and steps per RF period.
- Confirm the RustBCA table has useful statistics around the impact-energy range.
- Confirm the integrated source is plausible before increasing `nlaunch_total`.
- The output species are W, W+, W2+, W3+, and W4+; ADAS evolves among them.
- Treat RF sputtering as a reduced phase-average until a kinetic IEDF is supplied.
- The current surface areal-density curve is gross erosion; net deposition needs
  a pweight-aware wall-impact tally or the OpenEdge surface-composition model.
"""),
]

out = CASE_DIR / "scripts/analysis.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")

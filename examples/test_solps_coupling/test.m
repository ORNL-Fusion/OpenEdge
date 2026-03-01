clearvars;
close all;
clc;

addpath(genpath('/Users/42d/SOLPS-routines/Matlab'));
addpath(genpath('/Users/42d/SOLPS-routines'));

%% Inputs
run_path = '/Users/42d/solps_iter_oe/v0_1';
run_path_updated = '';  % set later when coupled SOLPS results exist
dump_path = '/Users/42d/OpenedgeGPU/examples/test_solps_coupling/output/case.dump';
equ_file = '/Users/42d/solps_iter_oe/baserun/g000001.00001_symm.X4.equ';
dt_oe = 1e-5;              % OpenEdge timestep [s]
dt_solps = 1e-5;           % SOLPS timestep [s] (b2mndr_dtim)
source_switch_nsteps = 10; % switch every N SOLPS steps (10*1e-5 = 100.d-6)
switch_guard_fraction = 0.1; % keep sources_time_switch slightly below boundary to avoid tim==tsw
analytical_mode = false;     % true: write analytical Gaussian sources for coupling test
analytical_only = false;     % true: stop after writing analytical source/profile files
analytical_total_steps = 110;


analytical_sources_subdir = 'analytical_sources';
analytical_peak_atoms_m3s = 1.0e25;  % Gaussian peak [atoms/m^3/s]
analytical_sig_r = 0.10;             % Gaussian sigma in R [m]
analytical_sig_z = 0.10;             % Gaussian sigma in Z [m]
analytical_r0_start = 3.565;         % moving center start
analytical_z0_start = -2.87;
analytical_r0_end = 2.628;           % moving center end
analytical_z0_end = -2.355;
analytical_write_gif = true;
analytical_show_live = true;       % show source frames on screen while writing
analytical_gif_name = 'source.gif';
analytical_gif_delay_s = 0.20;
analytical_gif_scale = 1.0e25;       % plotted value = source / scale
analytical_gif_clim = [0 1];
save_mat = true;
do_log_plot = false;
xlim_plot = [2.6 4.5];
ylim_plot = [-4 -2];
psi_core_n = 0.835;         % core contour from equilibrium file
outdir = fullfile(fileparts(dump_path), 'Figs');
fig_base = 'solps_coupling_overview_paper';
write_source2d_series = true;
n_source_files = 10;                               % requested coarse time slicing
sources_dir = fullfile(run_path, 'SOURCES');      % SOLPS source folder (dump-mapped mode)
deploy_to_run_root = true;                        % copy generated source/profile files into run_path
source_prefix = 'source2d.';
source_digits = 5;                                 % source2d.00001 style (avoid mixed naming)
source_scale_factor = 1.0;                        % multiply source values before writing source2d.*
clean_old_sources = true;                          % remove stale source/profile files before writing
write_b2_profiles = true;                          % write b2.sources.profile chain
profile_base = 'b2.sources.profile';
source_species_idx = 14 + 2;                       % Li+1 is species 14 in SOLPS 0-based indexing

ns_species = [];                                    % [] -> auto from Case.State.ns

% Paper-style plotting
font_all = 14;
title_font = 14;
line_w = 1.4;
axis_w = 1.2;
grid_alpha = 0.22;
wall_w = 2.6;
core_w = 2.8;
wall_col = [0 0 0];
core_col = [0.12 0.55 0.16];

AMU_KG = 1.66053906660e-27;
LI_AMU = 6.94;
LI_ATOM_MASS_KG = LI_AMU * AMU_KG;
LI_RHO = 534;              % kg/m^3 (liquid Li, nominal)
DROPLET_R0 = 2.5e-3;       % m (2.5 mm)

%% Load SOLPS case
Case = load_solps_case(run_path, 1, "none", "none", "none");
fprintf('Loaded SOLPS case: %s\n', run_path);
fprintf('Grid size with guard cells: nx+2=%d, ny+2=%d\n', Case.Geo.nx + 2, Case.Geo.ny + 2);
[rSep, zSep, rSep2, zSep2] = get_solps_sep_curves(Case.Geo);
[rCoreEq, zCoreEq] = get_core_from_equ_file(equ_file, psi_core_n, rSep, zSep);

%% Optional analytical Gaussian source mode (for coupling smoke test)
if analytical_mode
    if isempty(ns_species)
        if isfield(Case, 'State') && isfield(Case.State, 'ns') && ~isempty(Case.State.ns)
            ns_use = Case.State.ns;
        else
            ns_use = 17;
        end
    else
        ns_use = ns_species;
    end
    if source_species_idx < 1 || source_species_idx > ns_use
        error('source_species_idx=%d must be within [1, %d].', source_species_idx, ns_use);
    end

    n_windows = max(1, ceil(double(analytical_total_steps) / double(source_switch_nsteps)));
    sources_dir = fullfile(run_path, analytical_sources_subdir);
    write_analytical_gaussian_sources(Case, sources_dir, source_prefix, source_digits, ...
        profile_base, dt_solps, source_switch_nsteps, switch_guard_fraction, ...
        analytical_total_steps, n_windows, ns_use, source_species_idx, ...
        analytical_peak_atoms_m3s, analytical_r0_start, analytical_z0_start, ...
        analytical_r0_end, analytical_z0_end, analytical_sig_r, analytical_sig_z, ...
        clean_old_sources, write_b2_profiles, ...
        analytical_show_live, ...
        analytical_write_gif, analytical_gif_name, analytical_gif_delay_s, ...
        analytical_gif_scale, analytical_gif_clim);

    if deploy_to_run_root
        copy_generated_sources_to_run_root(sources_dir, run_path);
    end
    if analytical_only
        fprintf('Analytical-only source generation complete.\n');
        return;
    end
end

%% Read dump (R,Z,id,type,pmass,timestep)
[R, Z, ts, ids, types, pmass] = read_case_dump_full(dump_path);
fprintf('Read dump points: %d\n', numel(R));
fprintf('Timestep range: %d -> %d\n', min(ts), max(ts));
ts_unique = unique(ts);
fprintf('Number of timesteps in dump: %d\n', numel(ts_unique));
mass_loss_pct = compute_mass_loss_percent_per_row(ids, ts, pmass);

%% Compute evaporated mass from per-particle mass decrease
dm_kg = compute_mass_loss_per_row(ids, ts, pmass);   % kg lost at each row location
dn_atoms = dm_kg / LI_ATOM_MASS_KG;                  % atoms lost at each row location
dn_atoms = max(dn_atoms, 0);
fprintf('Total evaporated Li atoms in dump: %.6e\n', sum(dn_atoms));
V0 = (4.0/3.0) * pi * DROPLET_R0^3;
m0 = LI_RHO * V0;
N0 = m0 / LI_ATOM_MASS_KG;
frac_evap = sum(dn_atoms) / max(N0, eps);
fprintf('2.5 mm droplet initial atoms (rho=%.1f kg/m^3): %.6e\n', LI_RHO, N0);
fprintf('Evaporated fraction vs 2.5 mm initial inventory: %.2f %%\n', 100 * frac_evap);

%% Map row locations to SOLPS cells
[I1, ierr] = point2cell_solps_1d(Case.Geo, R, Z);
I1 = I1(:); ierr = ierr(:);
ok = (ierr == 0) & ~isnan(I1);
fprintf('Mapped points: %d / %d (%.2f%%)\n', nnz(ok), numel(I1), 100.0 * nnz(ok) / max(1, numel(I1)));

%% Build SOLPS-cell fields
nCells = (Case.Geo.nx + 2) * (Case.Geo.ny + 2);

% 1) Occupancy (for reference)
idx_occ = I1(ok); idx_occ = idx_occ(:);
occ_1d = accumarray(idx_occ, ones(size(idx_occ)), [nCells,1], @sum, 0);
occ_2d = reshape(occ_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);

% 2) Evaporated atoms integrated over simulation
idx_src = I1(ok); idx_src = idx_src(:);
vals_src = dn_atoms(ok); vals_src = vals_src(:);
atoms_int_1d = accumarray(idx_src, vals_src, [nCells,1], @sum, 0);
atoms_int_2d = reshape(atoms_int_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);

% 3) Time-averaged source rate [atoms/s per cell]
t_total = max((max(ts) - min(ts)) * dt_oe, eps);
atoms_rate_2d = atoms_int_2d / t_total;
% 4) Volumetric source [atoms/m^3/s] using SOLPS cell volumes
vol = Case.Geo.vol;
source_m3s_2d = zeros(size(atoms_rate_2d));
okv = isfinite(vol) & (vol > 0);
source_m3s_2d(okv) = atoms_rate_2d(okv) ./ vol(okv);

fprintf('Nonzero source cells: %d\n', nnz(atoms_rate_2d > 0));
fprintf('Peak source: %.6e atoms/m^3/s\n', max(source_m3s_2d(:)));

%% Write source2d.* series for SOLPS (piecewise-constant in time)
if write_source2d_series
    if isempty(ns_species)
        if isfield(Case, 'State') && isfield(Case.State, 'ns') && ~isempty(Case.State.ns)
            ns_use = Case.State.ns;
        else
            ns_use = 17;
        end
    else
        ns_use = ns_species;
    end
    if source_species_idx < 1 || source_species_idx > ns_use
        error('source_species_idx=%d must be within [1, %d].', source_species_idx, ns_use);
    end

    ts_min = min(ts);
    ts_max = max(ts);
    n_ts_total = ts_max - ts_min + 1;
    n_bins = max(1, min(n_source_files, n_ts_total));
    edges = floor(linspace(double(ts_min), double(ts_max) + 1.0, n_bins + 1));
    edges(1) = ts_min;
    edges(end) = ts_max + 1;
    for ib = 2:numel(edges)
        if edges(ib) <= edges(ib-1)
            edges(ib) = edges(ib-1) + 1;
        end
    end
    edges(end) = ts_max + 1;

    if ~exist(sources_dir, 'dir')
        mkdir(sources_dir);
    end
    if clean_old_sources
        delete(fullfile(sources_dir, 'source2d.*'));
        delete(fullfile(sources_dir, 'b2.sources.profile*'));
    end
    tw_file = fullfile(sources_dir, 'sources_time_windows.txt');
    fid_tw = fopen(tw_file, 'w');
    fprintf(fid_tw, '# idx  filename         t_start[s]      t_end[s]        dt_bin[s]      total_atoms/s   switch_time_solps[s]\n');
    src_names = strings(n_bins,1);
    t_switch = zeros(n_bins,1);
    src_total = zeros(n_bins,1);
    src_max = zeros(n_bins,1);
    src_nz = zeros(n_bins,1);

    fprintf('Writing %d source2d files to %s (scale factor = %.6g)\n', n_bins, sources_dir, source_scale_factor);
    for ib = 1:n_bins
        t0 = edges(ib);
        t1 = edges(ib+1);
        sel_bin = ok & (ts >= t0) & (ts < t1);
        idx_bin = I1(sel_bin);
        val_bin = dn_atoms(sel_bin);
        atoms_int_bin_1d = accumarray(idx_bin(:), val_bin(:), [nCells, 1], @sum, 0);
        atoms_int_bin_2d = reshape(atoms_int_bin_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);
        dt_bin = max((t1 - t0) * dt_oe, eps);
        atoms_rate_bin_2d = atoms_int_bin_2d / dt_bin;  % atoms/s per SOLPS cell
        atoms_rate_bin_2d = source_scale_factor * atoms_rate_bin_2d;

        src_names(ib) = sprintf('%s%0*d', source_prefix, source_digits, ib);
        out_src = fullfile(sources_dir, src_names(ib));
        tot_atoms_s = write_source2d_file(out_src, atoms_rate_bin_2d, ns_use, source_species_idx);
        src_total(ib) = tot_atoms_s;
        src_max(ib) = max(atoms_rate_bin_2d(:), [], 'omitnan');
        src_nz(ib) = nnz(atoms_rate_bin_2d > 0);
        % Absolute switch times on SOLPS clock; subtract a small guard so
        % each new switch is strictly greater than current tim at profile read.
        t_boundary = ib * source_switch_nsteps * dt_solps;
        t_guard = max(eps(t_boundary), switch_guard_fraction * dt_solps);
        t_switch(ib) = max(0.0, t_boundary - t_guard);
        fprintf(fid_tw, '%3d  %-16s %.8e  %.8e  %.8e  %.8e  %.8e\n', ...
            ib, src_names(ib), ...
            (t0 - ts_min) * dt_oe, (t1 - ts_min) * dt_oe, dt_bin, tot_atoms_s, t_switch(ib));
    end
    fclose(fid_tw);
    fprintf('Saved source timing table: %s\n', tw_file);
    if any(diff(t_switch) <= 0)
        error('Non-increasing t_switch generated. Check source_switch_nsteps and dt_solps.');
    end
    fprintf('Source summary (idx file switch[s] total_atoms/s max_cell_atoms/s nonzero_cells)\n');
    for ib = 1:n_bins
        fprintf('%3d  %-14s  %.8e  %.8e  %.8e  %d\n', ...
            ib, src_names(ib), t_switch(ib), src_total(ib), src_max(ib), src_nz(ib));
    end
    if write_b2_profiles
        write_b2_profile_chain(sources_dir, profile_base, src_names, t_switch);
    end
    if deploy_to_run_root
        copy_generated_sources_to_run_root(sources_dir, run_path);
    end
end

%% Combined 3-panel figure
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
fig = figure('Color', 'w', 'Position', [80 80 1750 560]);
tlo = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: OpenEdge trajectories
ax1 = nexttile(tlo, 1);
hold(ax1, 'on');
ut = unique(types);
mk = {'o','s','^','d','v','>','<','p','h'};
% Wall/PFC geometry from SOLPS mesh.extra (same as Jeremy's plotting)
meshExtra = plot_mesh_extra(run_path,0,1);
if ~isempty(meshExtra) && isfield(meshExtra,'r') && isfield(meshExtra,'z')
    for iL = 1:size(meshExtra.r,2)
        plot(ax1, meshExtra.r(:,iL), meshExtra.z(:,iL), '-', 'Color', wall_col, 'LineWidth', wall_w);
    end
end
% Separatrix from SOLPS grid
hSep = gobjects(0);
if ~isempty(rSep)
    hSep = plot(ax1, rSep, zSep, '-', 'Color', [0.85 0.10 0.10], 'LineWidth', line_w + 0.4);
end
if ~isempty(rSep2)
    plot(ax1, rSep2, zSep2, '--', 'Color', [0.75 0.15 0.15], 'LineWidth', line_w);
end
% Core curve from equilibrium file (psi_N contour)
hPsi = gobjects(0);
if ~isempty(rCoreEq)
    hPsi = plot(ax1, rCoreEq, zCoreEq, '-', 'Color', core_col, 'LineWidth', core_w);
end
hType = gobjects(numel(ut),1);
for it = 1:numel(ut)
    sel = (types == ut(it));
    this_m = mk{mod(it-1, numel(mk)) + 1};
    hType(it) = scatter(ax1, R(sel), Z(sel), 22, mass_loss_pct(sel), this_m, ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.95, ...
        'DisplayName', sprintf('type %d', ut(it)));
end

% single representative handle for legend label "droplet"
hDrop = hType(find(isgraphics(hType), 1, 'first'));
xlabel(ax1, 'R (m)');
ylabel(ax1, 'Z (m)');
title(ax1, '1) OpenEdge trajectories colored by mass loss (%)', 'FontSize', title_font);
grid(ax1, 'on');
axis(ax1, 'equal');
xlim(ax1, xlim_plot);
ylim(ax1, ylim_plot);
colormap(ax1, parula);
ml = mass_loss_pct(isfinite(mass_loss_pct));
if isempty(ml)
    caxis(ax1, [0 1]);
else
    cmax = max(5, prctile(ml, 99.5));
    caxis(ax1, [0 cmax]);
end
cb1 = colorbar(ax1);
cb1.Label.String = 'Droplet mass loss (%)';
cb1.FontSize = font_all;
cb1.Label.FontSize = font_all;
leg_h = gobjects(0);
leg_t = {};
if ~isempty(hSep) && isgraphics(hSep)
    leg_h(end+1) = hSep; %#ok<AGROW>
    leg_t{end+1} = 'separatrix'; %#ok<AGROW>
end
if ~isempty(hDrop) && isgraphics(hDrop)
    leg_h(end+1) = hDrop; %#ok<AGROW>
    leg_t{end+1} = 'droplet'; %#ok<AGROW>
end
if ~isempty(hPsi) && isgraphics(hPsi)
    leg_h(end+1) = hPsi; %#ok<AGROW>
    leg_t{end+1} = 'core'; %#ok<AGROW>
end
if ~isempty(leg_h)
    lg1 = legend(ax1, leg_h, leg_t, 'Location', 'best');
    lg1.FontSize = font_all - 1;
end
hold(ax1, 'off');

% Panel 2: mapped source on SOLPS grid
ax2 = nexttile(tlo, 2);
if do_log_plot
    data2 = log10(source_m3s_2d + 1);
    ttl2 = '2) log_{10}(Li source + 1) on SOLPS grid';
else
    data2 = source_m3s_2d;
    ttl2 = '2) Li source (atoms/m^3/s) on SOLPS grid';
end
plot_solps_field_on_axes(ax2, Case.Geo, data2, ttl2, xlim_plot, ylim_plot, true);

% Panel 3: placeholder for coupled SOLPS results
ax3 = nexttile(tlo, 3);
if ~isempty(run_path_updated) && isfolder(run_path_updated)
    CaseUpd = load_solps_case(run_path_updated, 1, "none", "none", "none");
    data3 = CaseUpd.State.ne;
    plot_solps_field_on_axes(ax3, CaseUpd.Geo, data3, '3) Coupled SOLPS result placeholder: n_e', xlim_plot, ylim_plot, true);
else
    axis(ax3, [xlim_plot ylim_plot]);
    axis(ax3, 'equal');
    grid(ax3, 'on');
    xlabel(ax3, 'R (m)');
    ylabel(ax3, 'Z (m)');
    title(ax3, '3) Placeholder: coupled SOLPS result', 'FontSize', title_font);
    text(ax3, mean(xlim_plot), mean(ylim_plot), ...
        sprintf('Set run_path_updated\nto plot coupled result'), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', font_all - 1, 'Color', [0.2 0.2 0.2]);
end

% Harmonize style across all panels
style_axes(ax1, font_all, axis_w, grid_alpha);
style_axes(ax2, font_all, axis_w, grid_alpha);
style_axes(ax3, font_all, axis_w, grid_alpha);

% Save paper-ready exports
png_file = fullfile(outdir, [fig_base '.png']);
pdf_file = fullfile(outdir, [fig_base '.pdf']);
exportgraphics(fig, png_file, 'Resolution', 300);
exportgraphics(fig, pdf_file, 'ContentType', 'vector');
fprintf('Saved figure PNG: %s\n', png_file);
fprintf('Saved figure PDF: %s\n', pdf_file);

if save_mat
    out_mat = fullfile(fileparts(dump_path), 'mapped_case_dump_to_solps.mat');
    save(out_mat, 'occ_2d', 'atoms_int_2d', 'atoms_rate_2d', 'source_m3s_2d', 'dt_oe', 't_total');
    fprintf('Saved mapping arrays: %s\n', out_mat);
end

function plot_solps_field_on_axes(ax, Geo, data2d, ttl, xlim_plot, ylim_plot, add_colorbar)
% Render SOLPS quadrilateral cells directly on provided axes.
% Geo.pr / Geo.pz are 4 x nCells (ordered vertices).
vals = data2d(:);
pp = patch(ax, Geo.pr, Geo.pz, vals', ...
    'FaceColor', 'flat', ...
    'EdgeColor', [0.45 0.45 0.45], ...
    'LineWidth', 0.08);
set(ax, 'Layer', 'top');
axis(ax, 'equal');
xlim(ax, xlim_plot);
ylim(ax, ylim_plot);
grid(ax, 'on');
xlabel(ax, 'R (m)');
ylabel(ax, 'Z (m)');
title(ax, ttl);
if add_colorbar
    cb = colorbar(ax);
    cb.Label.String = 'value';
end
colormap(ax, turbo);
if ~all(~isfinite(vals))
    caxis(ax, [min(vals(isfinite(vals))) max(vals(isfinite(vals))) + eps]);
end
end


function [R, Z, ts, ids, types, pmass] = read_case_dump_full(fname)
% Read LAMMPS-like dump and return x->R, y->Z plus id/type/v_pmass.

fid = fopen(fname, 'r');
if fid < 0
    error('Could not open dump file: %s', fname);
end

R = [];
Z = [];
ts = [];
ids = [];
types = [];
pmass = [];

this_ts = NaN;
this_n = NaN;

while true
    tline = fgetl(fid);
    if ~ischar(tline)
        break;
    end
    tline = strtrim(tline);

    if strcmp(tline, 'ITEM: TIMESTEP')
        this_ts = str2double(strtrim(fgetl(fid)));
    elseif strcmp(tline, 'ITEM: NUMBER OF ATOMS')
        this_n = str2double(strtrim(fgetl(fid)));
    elseif startsWith(tline, 'ITEM: BOX BOUNDS')
        fgetl(fid); fgetl(fid); fgetl(fid);
    elseif startsWith(tline, 'ITEM: ATOMS')
        parts = split(tline);
        headers = parts(3:end);
        iid = find(strcmp(headers, 'id'), 1);
        itype = find(strcmp(headers, 'type'), 1);
        ix = find(strcmp(headers, 'x'), 1);
        iy = find(strcmp(headers, 'y'), 1);
        im = find(strcmp(headers, 'v_pmass'), 1);
        if isempty(iid) || isempty(ix) || isempty(iy)
            error('ATOMS fields must include id, x, y.');
        end
        if isempty(itype)
            itype = [];
        end
        if isempty(im)
            im = [];
        end
        if isnan(this_n)
            error('NUMBER OF ATOMS missing before ATOMS block.');
        end

        for k = 1:this_n
            row = strtrim(fgetl(fid));
            vals = split(row);
            ids(end+1,1) = str2double(vals{iid}); %#ok<AGROW>
            if ~isempty(itype)
                types(end+1,1) = str2double(vals{itype}); %#ok<AGROW>
            else
                types(end+1,1) = 1; %#ok<AGROW>
            end
            R(end+1,1) = str2double(vals{ix}); %#ok<AGROW>
            Z(end+1,1) = str2double(vals{iy}); %#ok<AGROW>
            if ~isempty(im)
                pmass(end+1,1) = str2double(vals{im}); %#ok<AGROW>
            else
                pmass(end+1,1) = NaN; %#ok<AGROW>
            end
            ts(end+1,1) = this_ts; %#ok<AGROW>
        end
    end
end

fclose(fid);
end


function dm_kg = compute_mass_loss_per_row(ids, ts, pmass)
% dm_kg(i): evaporated mass assigned to sample i.
% For each particle id, dm at row k is max(m_k - m_{k+1}, 0).

N = numel(ids);
dm_kg = zeros(N,1);

if all(~isfinite(pmass))
    warning('v_pmass not found/finite in dump. dm_kg set to zero.');
    return;
end

[~, ord] = sortrows([ids(:), ts(:)], [1 2]);
ids_s = ids(ord);
pm_s = pmass(ord);

u = unique(ids_s);

for i = 1:numel(u)
    sel = find(ids_s == u(i));
    if numel(sel) < 2
        continue;
    end
    m0 = pm_s(sel(1:end-1));
    m1 = pm_s(sel(2:end));
    dml = m0 - m1;
    dml(~isfinite(dml)) = 0;
    dml = max(dml, 0);
    dm_kg(ord(sel(1:end-1))) = dml;
end
end

function pct = compute_mass_loss_percent_per_row(ids, ts, pmass)
% mass-loss percentage referenced to initial pmass for each particle id
N = numel(ids);
pct = nan(N,1);
if all(~isfinite(pmass))
    warning('v_pmass not found/finite in dump. mass loss percent is NaN.');
    return;
end

[~, ord] = sortrows([ids(:), ts(:)], [1 2]);
ids_s = ids(ord);
pm_s = pmass(ord);
u = unique(ids_s);

for i = 1:numel(u)
    sel = find(ids_s == u(i));
    if isempty(sel)
        continue;
    end
    m_track = pm_s(sel);
    i0 = find(isfinite(m_track) & (m_track > 0), 1, 'first');
    if isempty(i0)
        continue;
    end
    m0 = m_track(i0);
    p = 100.0 * (1.0 - m_track / m0);
    p(~isfinite(p)) = nan;
    p = max(p, 0.0);
    p = min(p, 100.0);
    pct(ord(sel)) = p;
end
end

function [rSep, zSep, rSep2, zSep2] = get_solps_sep_curves(Geo)
% Use SOLPS-native topology-aware separatrix curves.
rSep = [];
zSep = [];
rSep2 = [];
zSep2 = [];

if ~isfield(Geo, 'Sep') || isempty(Geo.Sep) || ~isfield(Geo,'jsep')
    return;
end
if isfield(Geo.Sep, 'r') && isfield(Geo.Sep, 'z')
    rSep = Geo.Sep.r(:);
    zSep = Geo.Sep.z(:);
end
if isfield(Geo.Sep, 'r2') && isfield(Geo.Sep, 'z2')
    rSep2 = Geo.Sep.r2(:);
    zSep2 = Geo.Sep.z2(:);
end
end

function [rCore, zCore] = get_core_from_equ_file(equ_file, psi_core_n, rSep, zSep)
% Parse .equ text file directly (same style as Python readEquilibrium),
% then extract one clean psi_N core contour.
rCore = [];
zCore = [];
if ~isfile(equ_file)
    return;
end

[r, z, psi_delta] = parse_equilibrium_ascii(equ_file);
if isempty(r) || isempty(z) || isempty(psi_delta)
    return;
end

% For files containing (psi - psib): separatrix is near 0, axis is min.
psi_axis = min(psi_delta(:), [], 'omitnan');
psi_sep = 0.0;
if nargin >= 4 && ~isempty(rSep) && ~isempty(zSep)
    [RR, ZZ] = meshgrid(r, z);
    psi_sep_s = interp2(RR, ZZ, psi_delta, rSep(:), zSep(:), 'linear', nan);
    if any(isfinite(psi_sep_s))
        psi_sep = median(psi_sep_s(isfinite(psi_sep_s)));
    end
end
den = psi_sep - psi_axis;
if ~isfinite(den) || abs(den) <= eps
    return;
end

psiN = (psi_delta - psi_axis) ./ den;
C = contourc(r, z, psiN, [psi_core_n psi_core_n]);
[rCore, zCore] = contourc_longest_segment(C);
end

function [r, z, psi] = parse_equilibrium_ascii(equ_file)
% Reads jm/km, r(1:jm), z(1:km), and ((psi(j,k)-psib,...)) blocks.
r = [];
z = [];
psi = [];
jm = [];
km = [];

fid = fopen(equ_file, 'r');
if fid < 0
    return;
end
c = onCleanup(@() fclose(fid));

read_r = false;
read_z = false;
read_psi = false;

while true
    tline = fgetl(fid);
    if ~ischar(tline)
        break;
    end
    tok = strsplit(strtrim(tline));
    if isempty(tok)
        continue;
    end

    if numel(tok) >= 3 && strcmp(tok{1}, 'jm') && strcmp(tok{2}, '=')
        jm = str2double(tok{3});
        continue;
    end
    if numel(tok) >= 3 && strcmp(tok{1}, 'km') && strcmp(tok{2}, '=')
        km = str2double(tok{3});
        continue;
    end

    if strcmp(tok{1}, '((psi(j,k)-psib,j=1,jm),k=1,km)')
        read_psi = true; read_z = false; read_r = false;
        continue;
    end
    if strcmp(tok{1}, 'z(1:km);')
        read_z = true; read_r = false; read_psi = false;
        continue;
    end
    if strcmp(tok{1}, 'r(1:jm);')
        read_r = true; read_z = false; read_psi = false;
        continue;
    end

    nums = sscanf(tline, '%f').';
    if isempty(nums)
        read_r = false; read_z = false; read_psi = false;
        continue;
    end
    if read_psi
        psi = [psi, nums]; %#ok<AGROW>
    elseif read_z
        z = [z, nums]; %#ok<AGROW>
    elseif read_r
        r = [r, nums]; %#ok<AGROW>
    end
end

if isempty(jm), jm = numel(r); end
if isempty(km), km = numel(z); end
if isempty(r) || isempty(z) || isempty(psi)
    r = []; z = []; psi = [];
    return;
end

jm = min(jm, numel(r));
km = min(km, numel(z));
r = r(1:jm);
z = z(1:km);
nneed = jm * km;
if numel(psi) < nneed
    r = []; z = []; psi = [];
    return;
end
psi = reshape(psi(1:nneed), [jm, km]).';
end

function [xseg, yseg] = contourc_longest_segment(C)
% Keep only the longest contour segment to avoid tiny dot-like artifacts.
xseg = [];
yseg = [];
if isempty(C)
    return;
end
k = 1;
best_len = -inf;
best_x = [];
best_y = [];
while k < size(C,2)
    n = C(2,k);
    if n <= 1
        break;
    end
    idx = (k+1):(k+n);
    if idx(end) > size(C,2)
        break;
    end
    x = C(1,idx).';
    y = C(2,idx).';
    seglen = sum(hypot(diff(x), diff(y)), 'omitnan');
    if seglen > best_len
        best_len = seglen;
        best_x = x;
        best_y = y;
    end
    k = k + n + 1;
end
if ~isempty(best_x)
    xseg = best_x;
    yseg = best_y;
end
end

function style_axes(ax, font_size, axis_w, grid_alpha)
set(ax, 'FontSize', font_size, ...
    'LineWidth', axis_w, ...
    'Box', 'on', ...
    'TickDir', 'out', ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on', ...
    'Layer', 'top');
ax.XAxis.FontSize = font_size;
ax.YAxis.FontSize = font_size;
ax.XLabel.FontSize = font_size;
ax.YLabel.FontSize = font_size;
if isgraphics(ax.Title)
    ax.Title.FontSize = font_size;
end
ax.GridAlpha = grid_alpha;
ax.MinorGridAlpha = max(0.10, grid_alpha * 0.8);
ax.XColor = [0.10 0.10 0.10];
ax.YColor = [0.10 0.10 0.10];
ax.TickLength = [0.014 0.014];
end

function total_atoms_s = write_source2d_file(out_file, atoms_rate_2d, ns, li_idx)
% SOLPS source2d writer:
% For each species block, write ny+2 lines of nx+2 values.
% Li species gets atoms/s per cell, others are zero.
src = max(atoms_rate_2d, 0.0);
[nxg, nyg] = size(src);
zrow = zeros(1, nxg);

fid = fopen(out_file, 'w');
if fid < 0
    error('Could not open source file for writing: %s', out_file);
end
c = onCleanup(@() fclose(fid));

for is = 1:ns
    for iy = 1:nyg
        if is == li_idx
            row = src(:, iy).';
        else
            row = zrow;
        end
        fprintf(fid, '%.8e', row(1));
        for ix = 2:nxg
            fprintf(fid, ' %.8e', row(ix));
        end
        fprintf(fid, '\n');
    end
end

total_atoms_s = sum(src(:), 'omitnan');
end

function write_b2_profile_chain(out_dir, profile_base, src_names, t_switch)
% Write:
%   b2.sources.profile
%   b2.sources.profile.2
%   ...
% with chained sources_filename and sources_time_switch.
n = numel(src_names);
if n < 1
    return;
end

for k = 2:n
    prof_name = sprintf('%s.%d', profile_base, k);
    prof_path = fullfile(out_dir, prof_name);
    fid = fopen(prof_path, 'w');
    if fid < 0
        error('Could not write profile file: %s', prof_path);
    end
    fprintf(fid, '&profile\n');
    fprintf(fid, 'read_sna0_2d=.true.\n');
    fprintf(fid, 'sna0_2d_filename="%s"\n', char(src_names(k)));

    if k < n
        next_prof = sprintf('%s.%d', profile_base, k + 1);
        fprintf(fid, 'sources_time_switch=%s\n', fmt_fortran_d_micro(t_switch(k)));
        fprintf(fid, 'sources_filename="%s"\n', next_prof);
    else
        % Keep final profile valid for SOLPS branches that require
        % sources_time_switch to advance at each profile read.
        fprintf(fid, 'sources_time_switch=1.d9\n');
    end
    fprintf(fid, '/\n');
    fclose(fid);
end

% Also write base file as entry-point alias of .1
base_path = fullfile(out_dir, profile_base);
fid = fopen(base_path, 'w');
if fid < 0
    error('Could not write profile file: %s', base_path);
end
fprintf(fid, '&profile\n');
fprintf(fid, 'read_sna0_2d=.true.\n');
fprintf(fid, 'sna0_2d_filename="%s"\n', char(src_names(1)));
if n > 1
    fprintf(fid, 'sources_time_switch=%s\n', fmt_fortran_d_micro(t_switch(1)));
    fprintf(fid, 'sources_filename="%s.2"\n', profile_base);
end
fprintf(fid, '/\n');
fclose(fid);

fprintf('Wrote %d profile files in %s\n', n, out_dir);
fprintf('Use in b2mn.dat: b2sraI_inputfile=''2'' and profile file %s\n', profile_base);
end

function s = fmt_fortran_d(x)
% Return Fortran D-format scalar string (e.g. 1.000000d-04).
s = lower(sprintf('%.6e', x));
s = strrep(s, 'e', 'd');
end

function s = fmt_fortran_d_micro(x)
% Prefer microsecond-style Fortran formatting used in SOLPS examples,
% e.g. 100.d-6, 200.d-6, ...
us = x / 1e-6;
if abs(us - round(us)) < 1e-9
    s = sprintf('%.0f.d-6', round(us));
else
    s = fmt_fortran_d(x);
end
end

function write_analytical_gaussian_sources(Case, out_dir, source_prefix, source_digits, ...
    profile_base, dt_solps, switch_nsteps, switch_guard_fraction, total_steps, n_windows, ...
    ns_use, li_idx, peak_m3s, r0_start, z0_start, r0_end, z0_end, sig_r, sig_z, ...
    clean_old_sources, write_b2_profiles, ...
    show_live, ...
    write_gif, gif_name, gif_delay_s, gif_scale, gif_clim)
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
if clean_old_sources
    delete(fullfile(out_dir, 'source2d.*'));
    delete(fullfile(out_dir, 'b2.sources.profile*'));
end

src_names = strings(n_windows,1);
t_switch = zeros(n_windows,1);
tw_file = fullfile(out_dir, 'sources_time_windows.txt');
fid_tw = fopen(tw_file, 'w');
if fid_tw < 0
    error('Could not open timing table for writing: %s', tw_file);
end
c = onCleanup(@() fclose(fid_tw)); %#ok<NASGU>
fprintf(fid_tw, '# idx  filename         step_start  step_end    t_start[s]      t_end[s]        total_atoms/s   switch_time_solps[s]\n');

gif_file = '';
if write_gif
    gif_file = fullfile(out_dir, gif_name);
    if exist(gif_file, 'file')
        delete(gif_file);
    end
end

for ib = 1:n_windows
    if n_windows == 1
        f = 0.0;
    else
        f = (ib - 1) / (n_windows - 1);
    end
    r0 = r0_start + f * (r0_end - r0_start);
    z0 = z0_start + f * (z0_end - z0_start);

    source_m3s = analytical_gaussian_on_solps_grid(Case, peak_m3s, r0, z0, sig_r, sig_z);
    atoms_rate_2d = max(source_m3s, 0.0) .* max(Case.Geo.vol, 0.0);

    src_names(ib) = sprintf('%s%0*d', source_prefix, source_digits, ib);
    out_src = fullfile(out_dir, src_names(ib));
    total_atoms_s = write_source2d_file(out_src, atoms_rate_2d, ns_use, li_idx);

    step_start = (ib - 1) * switch_nsteps;
    step_end = min(ib * switch_nsteps, total_steps);
    t_start = step_start * dt_solps;
    t_end = step_end * dt_solps;
    t_boundary = ib * switch_nsteps * dt_solps;
    t_guard = max(eps(t_boundary), switch_guard_fraction * dt_solps);
    t_switch(ib) = max(0.0, t_boundary - t_guard);

    fprintf(fid_tw, '%3d  %-16s %6d      %6d    %.8e  %.8e  %.8e  %.8e\n', ...
        ib, src_names(ib), step_start, step_end, t_start, t_end, total_atoms_s, t_switch(ib));

    if show_live || write_gif
        append_analytical_source_gif(ib, n_windows, source_m3s, Case, gif_file, ...
            gif_delay_s, gif_scale, gif_clim, step_start, step_end, show_live);
    end
end

if any(diff(t_switch) <= 0)
    error('Non-increasing t_switch generated in analytical mode.');
end
if write_b2_profiles
    write_b2_profile_chain(out_dir, profile_base, src_names, t_switch);
end
fprintf('Wrote %d analytical source files to %s\n', n_windows, out_dir);
fprintf('Source species index used (MATLAB 1-based): %d\n', li_idx);
if write_gif
    fprintf('Saved analytical source GIF: %s\n', gif_file);
end
end

function source = analytical_gaussian_on_solps_grid(Case, A, r0, z0, sig_r, sig_z)
R = Case.Geo.r2d_cen;
Z = Case.Geo.z2d_cen;
arg = ((R - r0).^2) ./ (2.0 * sig_r^2) + ((Z - z0).^2) ./ (2.0 * sig_z^2);
source = A .* exp(-arg);
source(~isfinite(source)) = 0.0;
end

function append_analytical_source_gif(iw, n_windows, source_m3s, Case, gif_file, ...
    gif_delay_s, gif_scale, gif_clim, step_start, step_end, show_live)
persistent fig_live ax_live patch_live sep_live sep2_live cb_live

if show_live
    if isempty(fig_live) || ~isvalid(fig_live)
        fig_live = figure('Color', 'w', 'Name', 'Analytical Gaussian Source', 'NumberTitle', 'off');
        ax_live = axes(fig_live);
        patch_live = patch(ax_live, Case.Geo.pr, Case.Geo.pz, source_m3s(:) ./ gif_scale, ...
            'FaceColor', 'flat', 'EdgeColor', 'none');
        hold(ax_live, 'on');
        [rSep, zSep, rSep2, zSep2] = get_solps_sep_curves(Case.Geo);
        sep_live = gobjects(0);
        sep2_live = gobjects(0);
        if ~isempty(rSep)
            sep_live = plot(ax_live, rSep, zSep, 'k-', 'LineWidth', 1.2);
        end
        if ~isempty(rSep2)
            sep2_live = plot(ax_live, rSep2, zSep2, 'k--', 'LineWidth', 1.0);
        end
        axis(ax_live, 'equal');
        axis(ax_live, 'tight');
        colormap(ax_live, turbo);
        cb_live = colorbar(ax_live);
    else
        set(patch_live, 'CData', source_m3s(:) ./ gif_scale);
        if ~isempty(sep_live) && ~isvalid(sep_live)
            sep_live = gobjects(0);
        end
        if ~isempty(sep2_live) && ~isvalid(sep2_live)
            sep2_live = gobjects(0);
        end
    end
    if ~isempty(cb_live) && isvalid(cb_live)
        cb_live.Label.String = sprintf('Li source / %.1e', gif_scale);
    end
    if numel(gif_clim) == 2
        set(ax_live, 'CLim', gif_clim);
    end
    xlabel(ax_live, 'R [m]');
    ylabel(ax_live, 'Z [m]');
    title(ax_live, sprintf('Analytical Gaussian source %d/%d (steps %d-%d)', ...
        iw, n_windows, step_start, step_end));
    drawnow;
end

fig = figure('Visible', 'off', 'Color', 'w');
ax = axes(fig);
patch(ax, Case.Geo.pr, Case.Geo.pz, source_m3s(:) ./ gif_scale, 'FaceColor', 'flat', 'EdgeColor', 'none');
hold(ax, 'on');
[rSep, zSep, rSep2, zSep2] = get_solps_sep_curves(Case.Geo);
if ~isempty(rSep)
    plot(ax, rSep, zSep, 'k-', 'LineWidth', 1.2);
end
if ~isempty(rSep2)
    plot(ax, rSep2, zSep2, 'k--', 'LineWidth', 1.0);
end
axis(ax, 'equal');
axis(ax, 'tight');
colormap(ax, turbo);
cb = colorbar(ax);
cb.Label.String = sprintf('Li source / %.1e', gif_scale);
if numel(gif_clim) == 2
    set(ax, 'CLim', gif_clim);
end
xlabel(ax, 'R [m]');
ylabel(ax, 'Z [m]');
title(ax, sprintf('Analytical Gaussian source %d/%d (steps %d-%d)', ...
    iw, n_windows, step_start, step_end));
drawnow;
frame = getframe(fig);
im = frame2im(frame);
close(fig);
[A, map] = rgb2ind(im, 256);
if ~isempty(gif_file)
    if iw == 1
        imwrite(A, map, gif_file, 'gif', 'LoopCount', Inf, 'DelayTime', gif_delay_s);
    else
        imwrite(A, map, gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', gif_delay_s);
    end
end
end

function copy_generated_sources_to_run_root(src_dir, run_dir)
if ~isfolder(run_dir)
    error('Run directory not found: %s', run_dir);
end
delete(fullfile(run_dir, 'source2d.*'));
delete(fullfile(run_dir, 'b2.sources.profile*'));
copyfile(fullfile(src_dir, 'source2d.*'), run_dir);
copyfile(fullfile(src_dir, 'b2.sources.profile*'), run_dir);
if exist(fullfile(src_dir, 'sources_time_windows.txt'), 'file')
    copyfile(fullfile(src_dir, 'sources_time_windows.txt'), run_dir);
end
fprintf('Deployed source2d.* and b2.sources.profile* from %s -> %s\n', src_dir, run_dir);
end



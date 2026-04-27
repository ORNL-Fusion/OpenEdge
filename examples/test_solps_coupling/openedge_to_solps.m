clearvars;
close all;
clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, 'matlab')));
addpath(genpath('/Users/42d/SOLPS-routines/Matlab'));
addpath(genpath('/Users/42d/SOLPS-routines'));

%% Inputs
run_path = '/Users/42d/solps_iter_oe/v0_1';
run_path_updated = '';  % set later when coupled SOLPS results exist
dump_path = '/Users/42d/OpenEdge/examples/test_solps_coupling/output/case.dump';        % optional: used for trajectory overview plot
mass_loss_path = '/Users/42d/OpenEdge/examples/test_solps_coupling/output/mass_loss.txt'; % source2d.* is built from this grid dump
equ_file = '/Users/42d/solps_iter_oe/baserun/g000001.00001_symm.X4.equ';
dt_oe = 1e-5;              % OpenEdge timestep [s]
dt_solps = 1e-5;           % SOLPS timestep [s] (b2mndr_dtim)
source_switch_nsteps = 1;  % switch every SOLPS step (preserve OE time evolution)
switch_guard_fraction = 0.1; % keep sources_time_switch slightly below boundary to avoid tim==tsw
target_source_duration_steps = 1000; % compress full OpenEdge source history into this many SOLPS steps ([] disables compression)
preserve_integrated_source = true;   % rescale source amplitude when compressing so each bin injects the same total atoms
save_mat = false;
do_log_plot = true;
xlim_plot = [2.6 4.5];
ylim_plot = [-4 -2];
psi_core_n = 0.835;         % core contour from equilibrium file
outdir = fullfile(fileparts(mass_loss_path), 'Figs');
fig_base = 'solps_coupling_overview_paper';
build_source_files = true;                         % true: regenerate source2d/profile files for the selected SOLPS run
write_source2d_series = build_source_files;
n_source_files = 10;                               % framework test: ~10 transient source windows over the nonzero pulse
sources_dir = fullfile(run_path, 'SOURCES');      % SOLPS source folder (dump-mapped mode)
deploy_to_run_root = build_source_files;          % copy generated source/profile files into run_path
source_prefix = 'source2d.';
source_digits = 5;                                 % source2d.00001 style (avoid mixed naming)
source_scale_factor = 1.0;                         % keep physical source magnitude
auto_scale_by_switch_dt = false;                   % disable extra amplification
clean_old_sources = true;                          % remove stale source/profile files before writing
write_b2_profiles = true;                          % write b2.sources.profile chain
profile_base = 'b2.sources.profile';
source_species_idx = 15;                           % MATLAB 1-based index for Li+1 (0-based species 14)

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

%% Read mass-loss grid dump and build transient source directly from it
nCells = (Case.Geo.nx + 2) * (Case.Geo.ny + 2);
ml_blocks = read_mass_loss_dump(mass_loss_path);
n_ml_blocks = numel(ml_blocks);
ml_ts = zeros(n_ml_blocks, 1);
source_m3s_blocks_1d = zeros(nCells, n_ml_blocks);
ml_method = '';
vol = Case.Geo.vol;
vol_1d = vol(:);
r_solps = Case.Geo.r2d_cen(:);
z_solps = Case.Geo.z2d_cen(:);
dR_oe = NaN;
dZ_oe = NaN;
vol_oe = [];

for ib = 1:n_ml_blocks
    ml_ts(ib) = ml_blocks(ib).timestep;
    [r_ml, z_ml, atoms_step_blk, method_blk] = extract_atoms_per_step_from_mass_loss_block(ml_blocks(ib), LI_ATOM_MASS_KG);
    if isempty(ml_method)
        ml_method = method_blk;
        dR_oe = infer_uniform_cell_size(r_ml);
        dZ_oe = infer_uniform_cell_size(z_ml);
        vol_oe = 2.0 * pi * r_ml * dR_oe * dZ_oe;
        fprintf('OpenEdge mass_loss grid spacing: dR=%.6e m, dZ=%.6e m\n', dR_oe, dZ_oe);
    end
    source_oe = zeros(size(atoms_step_blk));
    ok_oe = isfinite(vol_oe) & (vol_oe > 0);
    source_oe(ok_oe) = atoms_step_blk(ok_oe) ./ max(dt_oe * vol_oe(ok_oe), eps);
    source_m3s_blocks_1d(:, ib) = interpolate_openedge_source_to_solps( ...
        r_ml, z_ml, source_oe, r_solps, z_solps, vol_1d, sum(atoms_step_blk) / max(dt_oe, eps));
end

ml_block_step = infer_uniform_timestep_stride(ml_ts);
ml_block_dt_s = infer_dump_block_dt_seconds(ml_ts, dt_oe);
t_total = sum(ml_block_dt_s);
ml_t_edges_s = [0.0; cumsum(ml_block_dt_s(:))];
block_weights = ml_block_dt_s / max(t_total, eps);
source_m3s_mean_1d = source_m3s_blocks_1d * block_weights;
occ_1d = sum(source_m3s_blocks_1d > 0, 2);
occ_2d = reshape(occ_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);
atoms_rate_blocks_1d = source_m3s_blocks_1d .* vol_1d;
atoms_int_1d = atoms_rate_blocks_1d * ml_block_dt_s;
atoms_int_2d = reshape(atoms_int_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);
source_m3s_2d = reshape(source_m3s_mean_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);
atoms_rate_2d = source_m3s_2d .* vol;

fprintf('Read mass_loss blocks: %d\n', n_ml_blocks);
fprintf('mass_loss timestep range: %d -> %d\n', min(ml_ts), max(ml_ts));
fprintf('mass_loss stride: %d OpenEdge steps (%.6e s)\n', ml_block_step, ml_block_step * dt_oe);
fprintf('mass_loss conversion method: %s\n', ml_method);
fprintf('Total evaporated Li atoms from mass_loss: %.6e\n', sum(atoms_int_1d));
fprintf('Nonzero source cells: %d\n', nnz(atoms_rate_2d > 0));
fprintf('Peak source: %.6e atoms/m^3/s\n', max(source_m3s_2d(:)));
fprintf('Original source duration: %.6e s (%.1f SOLPS steps)\n', t_total, t_total / max(dt_solps, eps));

%% Optional trajectory dump for overview plotting only
have_dump_for_plot = exist(dump_path, 'file') == 2;
R = [];
Z = [];
ts = [];
ids = [];
types = [];
pmass = [];
mass_loss_pct = [];
if have_dump_for_plot
    [R, Z, ts, ids, types, pmass] = read_case_dump_full(dump_path);
    fprintf('Read dump points: %d\n', numel(R));
    fprintf('Dump timestep range: %d -> %d\n', min(ts), max(ts));
    ts_unique = unique(ts);
    fprintf('Number of timesteps in dump: %d\n', numel(ts_unique));
    mass_loss_pct = compute_mass_loss_percent_per_row(ids, ts, pmass);

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
else
    fprintf('Dump file not found, skipping trajectory panel: %s\n', dump_path);
end

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

    n_ts_total = n_ml_blocks;
    if n_source_files <= 0
        n_bins = n_ts_total;
    else
        n_bins = max(1, min(n_source_files, n_ts_total));
    end
    edges = floor(linspace(1.0, double(n_ml_blocks) + 1.0, n_bins + 1));
    edges(1) = 1;
    edges(end) = n_ml_blocks + 1;
    for ib = 2:numel(edges)
        if edges(ib) <= edges(ib-1)
            edges(ib) = edges(ib-1) + 1;
        end
    end
    edges(end) = n_ml_blocks + 1;

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

    switch_dt = max(source_switch_nsteps * dt_solps, eps);
    source_time_scale = 1.0;
    source_amplitude_scale = 1.0;
    target_source_duration_s = [];
    if ~isempty(target_source_duration_steps)
        target_source_duration_s = target_source_duration_steps * dt_solps;
        if ~(isfinite(target_source_duration_s) && target_source_duration_s > 0.0)
            error('target_source_duration_steps must be positive when provided.');
        end
        if target_source_duration_s < t_total
            source_time_scale = target_source_duration_s / max(t_total, eps);
            if preserve_integrated_source
                source_amplitude_scale = 1.0 / source_time_scale;
            end
        end
    end
    if auto_scale_by_switch_dt && (abs(source_time_scale - 1.0) > 1.0e-12)
        error('Disable auto_scale_by_switch_dt when using timeline compression.');
    end
    if abs(source_time_scale - 1.0) > 1.0e-12
        fprintf('Compressing source timeline: %.6e s -> %.6e s (%.1f -> %d SOLPS steps)\n', ...
            t_total, target_source_duration_s, t_total / max(dt_solps, eps), target_source_duration_steps);
        fprintf('Amplitude scale to preserve integrated atoms: %.6e\n', source_amplitude_scale);
    else
        fprintf('Source timeline compression disabled; using physical OpenEdge duration.\n');
    end
    fprintf('Writing %d source2d files to %s (base scale = %.6g, auto_scale_by_switch_dt=%d, switch_dt=%.8e s)\n', ...
        n_bins, sources_dir, source_scale_factor, auto_scale_by_switch_dt, switch_dt);
    for ib = 1:n_bins
        i0 = edges(ib);
        i1 = edges(ib+1) - 1;
        sel_bin = i0:i1;
        w_bin = ml_block_dt_s(sel_bin);
        w_sum = sum(w_bin);
        if w_sum <= 0
            w_bin = ones(size(w_bin));
            w_sum = sum(w_bin);
        end
        source_bin_1d = source_m3s_blocks_1d(:, sel_bin) * (w_bin / w_sum);
        atoms_rate_bin_2d = reshape(source_bin_1d .* vol_1d, [Case.Geo.nx + 2, Case.Geo.ny + 2]);  % atoms/s per SOLPS cell
        bin_scale = source_scale_factor * source_amplitude_scale;
        if auto_scale_by_switch_dt
            bin_scale = bin_scale * (w_sum / switch_dt);
        end
        atoms_rate_bin_2d = bin_scale * atoms_rate_bin_2d;

        src_names(ib) = sprintf('%s%0*d', source_prefix, source_digits, ib);
        out_src = fullfile(sources_dir, src_names(ib));
        tot_atoms_s = write_source2d_file(out_src, atoms_rate_bin_2d, ns_use, source_species_idx);
        src_total(ib) = tot_atoms_s;
        src_max(ib) = max(atoms_rate_bin_2d(:), [], 'omitnan');
        src_nz(ib) = nnz(atoms_rate_bin_2d > 0);
        t0_s_orig = ml_t_edges_s(i0);
        t1_s_orig = ml_t_edges_s(i1 + 1);
        t0_s = source_time_scale * t0_s_orig;
        t1_s = source_time_scale * t1_s_orig;
        dt_bin = max(t1_s - t0_s, eps);
        % Absolute switch times on SOLPS clock; subtract a small guard so each
        % new switch is strictly greater than current tim at profile read.
        t_boundary = t1_s;
        t_guard = max(eps(t_boundary), switch_guard_fraction * dt_solps);
        t_switch(ib) = max(0.0, t_boundary - t_guard);
        fprintf(fid_tw, '%3d  %-16s %.8e  %.8e  %.8e  %.8e  %.8e\n', ...
            ib, src_names(ib), ...
            t0_s, t1_s, dt_bin, tot_atoms_s, t_switch(ib));
    end
    fclose(fid_tw);
    fprintf('Saved source timing table: %s\n', tw_file);
    if any(diff(t_switch) <= 0)
        error('Non-increasing t_switch generated. Check source_switch_nsteps and dt_solps.');
    end
    fprintf('Source summary (idx file switch[s] switch[steps] total_atoms/s max_cell_atoms/s nonzero_cells)\n');
    for ib = 1:n_bins
        fprintf('%3d  %-14s  %.8e  %.1f steps  %.8e  %.8e  %d\n', ...
            ib, src_names(ib), t_switch(ib), t_switch(ib) / max(dt_solps, eps), src_total(ib), src_max(ib), src_nz(ib));
    end
    if write_b2_profiles
        write_b2_profile_chain(sources_dir, profile_base, src_names, t_switch);
    end
    if deploy_to_run_root
        copy_generated_sources_to_run_root(sources_dir, run_path);
    end
else
    fprintf('Plot-only mode: skipping source2d.* and b2.sources.profile* generation.\n');
end

%% Combined figure
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
fig = figure('Color', 'w', 'Position', [80 80 1180 560]);
tlo = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: OpenEdge trajectory overview
% Temporarily disabled to keep the framework script focused on source generation
% from mass_loss.txt. Restore the old trajectory panel here if needed later.

% Panel 2: mapped source on SOLPS grid
ax2 = nexttile(tlo, 1);
if do_log_plot
    data2 = log10(source_m3s_2d + 1);
    ttl2 = '1) log_{10}(Li source + 1) on SOLPS grid';
else
    data2 = source_m3s_2d;
    ttl2 = '1) Li source (atoms/m^3/s) on SOLPS grid';
end
plot_solps_field_on_axes(ax2, Case.Geo, data2, ttl2, xlim_plot, ylim_plot, true);

% Panel 3: placeholder for coupled SOLPS results
ax3 = nexttile(tlo, 2);
if ~isempty(run_path_updated) && isfolder(run_path_updated)
    CaseUpd = load_solps_case(run_path_updated, 1, "none", "none", "none");
    data3 = CaseUpd.State.ne;
    plot_solps_field_on_axes(ax3, CaseUpd.Geo, data3, '2) Coupled SOLPS result placeholder: n_e', xlim_plot, ylim_plot, true);
else
    axis(ax3, [xlim_plot ylim_plot]);
    axis(ax3, 'equal');
    grid(ax3, 'on');
    xlabel(ax3, 'R (m)');
    ylabel(ax3, 'Z (m)');
    title(ax3, '2) Placeholder: coupled SOLPS result', 'FontSize', title_font);
    text(ax3, mean(xlim_plot), mean(ylim_plot), ...
        sprintf('Set run_path_updated\nto plot coupled result'), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', font_all - 1, 'Color', [0.2 0.2 0.2]);
end

% Harmonize style across active panels
style_axes(ax2, font_all, axis_w, grid_alpha);
style_axes(ax3, font_all, axis_w, grid_alpha);

% Save paper-ready exports (robust across MATLAB graphics backends)
png_file = fullfile(outdir, [fig_base '.png']);
pdf_file = fullfile(outdir, [fig_base '.pdf']);
drawnow;
if ~isgraphics(fig)
    warning('Figure handle became invalid before export; skipping figure export.');
else
    export_target = fig;
    if exist('tlo', 'var') && isgraphics(tlo)
        export_target = tlo;
    end
    exportgraphics(export_target, png_file, 'Resolution', 300);
    fprintf('Saved figure PNG: %s\n', png_file);
    try
        exportgraphics(export_target, pdf_file, 'ContentType', 'vector');
        fprintf('Saved figure PDF: %s\n', pdf_file);
    catch ME_pdf
        warning('exportgraphics PDF failed (%s). Falling back to print -dpdf.', ME_pdf.message);
        print(fig, pdf_file, '-dpdf', '-painters');
        fprintf('Saved figure PDF (print fallback): %s\n', pdf_file);
    end
end

if save_mat
    out_mat = fullfile(fileparts(mass_loss_path), 'mapped_mass_loss_to_solps.mat');
    save(out_mat, 'occ_2d', 'atoms_int_2d', 'atoms_rate_2d', 'source_m3s_2d', ...
        'dt_oe', 't_total', 'ml_ts', 'ml_block_dt_s', 'ml_method');
    fprintf('Saved mapping arrays: %s\n', out_mat);
end

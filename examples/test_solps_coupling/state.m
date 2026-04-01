clearvars;
close all;
clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, 'matlab')));
addpath(genpath('/Users/42d/SOLPS-routines/Matlab'));
addpath(genpath('/Users/42d/SOLPS-routines'));

run_paths = {
    '/Users/42d/solps_iter_oe/v0_1/b2mn.exe.dir/plasmastate.0000'
    '/Users/42d/solps_iter_oe/v0_1/b2mn.exe.dir/plasmastate.0006'
    '/Users/42d/solps_iter_oe/v0_1/b2mn.exe.dir/plasmastate.0013'
};

state_source_mode = "plasmastate";  % "plasmastate" or "run_state"
snapshot_name = "latest";           % used when a run_paths entry is a directory
snapshot_subdir = "b2mn.exe.dir";   % used when run_paths point at the run root

show_openedge_panel = true;
openedge_color_mode = "cum_li_atoms";  % "mass_loss_pct" or "cum_li_atoms"
dump_path = fullfile(script_dir, 'output', 'case.dump');
mass_loss_path = fullfile(script_dir, 'output', 'mass_loss.txt');
wall_path = fullfile(script_dir, 'input', 'wall.surf');
dt_oe = 1.0e-5;

solps_quantity = "li_tot";  % "ne", "te_ev", "ti_ev", "po", "li_tot", or "li_frac"
solps_plot_mode = "auto";   % "auto", "linear", or "log10"
solps_clip_percentiles = [1.0 99.5];

% x_limits = [2.34 4.0];
 x_limits = [2.34 4.0];
 y_limits = [-3.8 1.1];
% y_limits = [-3.8 -2.1];
outdir = fullfile(script_dir, 'output', 'Figs');
fig_base = sprintf('openedge_solps_%s_overview', regexprep(char(solps_quantity), '[^A-Za-z0-9_]+', '_'));
export_dpi = 400;

n_cases = numel(run_paths);
Cases = cell(1, n_cases);
solps_fields = cell(1, n_cases);
panel_titles = cell(1, n_cases);

have_openedge_panel = show_openedge_panel && isfile(dump_path) && isfile(wall_path);
if show_openedge_panel && ~have_openedge_panel
    warning('OpenEdge panel skipped because dump/wall files were not both found.');
end

if have_openedge_panel && strcmpi(char(openedge_color_mode), 'cum_li_atoms') && ~isfile(mass_loss_path)
    warning('mass_loss.txt not found; falling back to mass_loss_pct coloring from case.dump.');
    openedge_color_mode = "mass_loss_pct";
end

OpenEdge = struct([]);
if have_openedge_panel
    OpenEdge = load_openedge_panel_data( ...
        dump_path, mass_loss_path, wall_path, dt_oe, openedge_color_mode);
end

cmin = inf;
cmax = -inf;

for i = 1:n_cases
    [case_run_path, snapshot_dir, snapshot_name_i] = resolve_case_paths(run_paths{i}, state_source_mode, snapshot_subdir);
    Cases{i} = load_case_for_plot(case_run_path, state_source_mode);

    snapshot_label = "";
    if strcmpi(char(state_source_mode), 'plasmastate')
        if strlength(snapshot_name_i) == 0
            snapshot_name_i = snapshot_name;
        end
        [Cases{i}.State, snapshot_label] = load_plasmastate_state( ...
            snapshot_dir, snapshot_name_i, Cases{i}.Geo, Cases{i}.State);
    end

    solps_fields{i} = get_state_field(Cases{i}, solps_quantity);
    solps_fields{i} = transform_solps_field_for_plot(solps_fields{i}, solps_quantity, solps_plot_mode);
    panel_titles{i} = get_panel_title(case_run_path);
    if strlength(snapshot_label) > 0
        panel_titles{i} = sprintf('%s\n%s', panel_titles{i}, char(snapshot_label));
    end

    data_i = solps_fields{i};
    finite_i = isfinite(data_i);
    if any(finite_i(:))
        cmin = min(cmin, min(data_i(finite_i)));
        cmax = max(cmax, max(data_i(finite_i)));
    end
end

[cmin, cmax] = compute_common_color_limits(solps_fields, solps_clip_percentiles, cmin, cmax);

[colorbar_label, ~] = get_solps_labels(solps_quantity, solps_plot_mode);

n_panels = n_cases + double(have_openedge_panel);
fig = figure('Color', 'w', 'Position', [100 100 max(1780, 430 * n_panels) 620]);
tl = tiledlayout(fig, 1, n_panels, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = gobjects(1, n_panels);
cb = gobjects(1, n_panels);

panel_idx = 1;
if have_openedge_panel
    SepOverlay = [];
    if ~isempty(Cases)
        SepOverlay = get_sep_from_SOLPS_grid(Cases{1}.Geo);
    end
    ax(panel_idx) = nexttile(tl);
    cb(panel_idx) = plot_openedge_panel(ax(panel_idx), OpenEdge, SepOverlay, x_limits, y_limits);
    set(ax(panel_idx), 'FontSize', 16, 'LineWidth', 1.5);
    if isgraphics(cb(panel_idx))
        cb(panel_idx).FontSize = 16;
        cb(panel_idx).Label.FontSize = 18;
        cb(panel_idx).Label.FontWeight = 'bold';
    end
    panel_idx = panel_idx + 1;
end

for i = 1:n_cases
    ax(panel_idx) = nexttile(tl);
    [ax(panel_idx), cb(panel_idx)] = plot_solps_2d_profile_a( ...
        Cases{i}, ...
        solps_fields{i}, ...
        panel_titles{i}, ...
        'ShowColorbar', true, ...
        'ColorbarLabel', colorbar_label, ...
        'ShowYLabel', panel_idx == 1, ...
        'ShowXLabel', true);
    caxis(ax(panel_idx), [cmin cmax]);
    % xlim(ax(panel_idx), x_limits);
    % ylim(ax(panel_idx), y_limits);
    colormap(ax(panel_idx), turbo);
    set(ax(panel_idx), 'FontSize', 16, 'LineWidth', 1.5);
    if isgraphics(cb(panel_idx))
        cb(panel_idx).FontSize = 16;
        cb(panel_idx).Label.FontSize = 18;
        cb(panel_idx).Label.FontWeight = 'bold';
    end
    panel_idx = panel_idx + 1;
end

valid_ax = ax(isgraphics(ax));
if numel(valid_ax) > 1
    linkaxes(valid_ax, 'xy');
end

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

png_file = fullfile(outdir, [fig_base '.png']);
pdf_file = fullfile(outdir, [fig_base '.pdf']);
drawnow;
if ~isgraphics(fig)
    warning('Figure handle became invalid before export; skipping figure export.');
else
    export_target = fig;
    if exist('tl', 'var') && isgraphics(tl)
        export_target = tl;
    end
    exportgraphics(export_target, png_file, 'Resolution', export_dpi);
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


function Case = load_case_for_plot(run_path, state_source_mode)
persistent cached_run_paths cached_cases

if isempty(cached_run_paths)
    cached_run_paths = {};
    cached_cases = {};
end

idx = find(strcmp(cached_run_paths, char(run_path)), 1, 'first');
if ~isempty(idx)
    Case = cached_cases{idx};
    return;
end

if strcmpi(char(state_source_mode), 'plasmastate')
    Case = struct();
    Case.run_path = run_path;
    Case.Geo = read_b2fgmtry(fullfile(run_path, 'b2fgmtry'), 0, 1);
    Case.g = [];
    Case.Tri = struct('geo', struct('nt', 0));
    Case.State = struct();
else
    Case = load_solps_case(run_path, 1, "none", "none", "none");
end

cached_run_paths{end + 1} = char(run_path); %#ok<AGROW>
cached_cases{end + 1} = Case; %#ok<AGROW>
end


function state_field = get_state_field(Case, quantity_mode)
switch lower(char(quantity_mode))
    case 'ne'
        state_field = Case.State.ne;
    case {'te', 'te_j'}
        state_field = Case.State.te;
    case 'te_ev'
        state_field = Case.State.te ./ electron_charge();
    case {'ti', 'ti_j'}
        state_field = Case.State.ti;
    case 'ti_ev'
        state_field = Case.State.ti ./ electron_charge();
    case 'po'
        state_field = Case.State.po;
    case {'li_tot', 'li_frac'}
        iLi = find((Case.State.zn == 3) & (abs(Case.State.am - 7) < 0.2));
        if isempty(iLi)
            error('Could not identify Li-7 in Case.State.');
        end

        Li_tot = sum(Case.State.na(:, :, iLi), 3);
        if strcmpi(char(quantity_mode), 'li_tot')
            state_field = Li_tot;
            return;
        end

        ne = Case.State.ne;
        state_field = zeros(size(Li_tot));
        valid = isfinite(Li_tot) & isfinite(ne) & (ne > 0);
        state_field(valid) = Li_tot(valid) ./ ne(valid);
    otherwise
        error('Unsupported solps_quantity: %s', char(quantity_mode));
end
end


function plot_field = transform_solps_field_for_plot(state_field, quantity_mode, plot_mode)
plot_field = state_field;

if ~should_use_log_plot(quantity_mode, plot_mode)
    return;
end

mask = isfinite(state_field) & (state_field > 0);
plot_field = nan(size(state_field));
plot_field(mask) = log10(state_field(mask));
end


function [cmin, cmax] = compute_common_color_limits(field_cell, clip_percentiles, cmin_in, cmax_in)
all_vals = [];
for i = 1:numel(field_cell)
    vals = field_cell{i};
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        all_vals = [all_vals; vals(:)]; %#ok<AGROW>
    end
end

if isempty(all_vals)
    cmin = 0.0;
    cmax = 1.0;
    return;
end

all_vals = sort(all_vals);
n = numel(all_vals);
i1 = max(1, min(n, round(clip_percentiles(1) / 100 * n)));
i2 = max(1, min(n, round(clip_percentiles(2) / 100 * n)));
cmin = all_vals(i1);
cmax = all_vals(i2);

if ~isfinite(cmin) || ~isfinite(cmax) || (cmax <= cmin)
    cmin = cmin_in;
    cmax = cmax_in;
end

if ~isfinite(cmin) || ~isfinite(cmax) || (cmax <= cmin)
    cmin = min(all_vals);
    cmax = max(all_vals);
end

if ~isfinite(cmin) || ~isfinite(cmax) || (cmax <= cmin)
    cmin = 0.0;
    cmax = 1.0;
end
end


function tf = should_use_log_plot(quantity_mode, plot_mode)
switch lower(char(plot_mode))
    case 'linear'
        tf = false;
        return;
    case 'log10'
        tf = true;
        return;
end

switch lower(char(quantity_mode))
    case {'ne', 'li_tot', 'li_frac'}
        tf = true;
    otherwise
        tf = false;
end
end


function [colorbar_label, figure_title] = get_solps_labels(quantity_mode, plot_mode)
switch lower(char(quantity_mode))
    case 'ne'
        colorbar_label = 'n_e (m^{-3})';
        figure_title = 'SOLPS Electron Density';
    case {'te', 'te_j'}
        colorbar_label = 'T_e (J)';
        figure_title = 'SOLPS Electron Temperature';
    case 'te_ev'
        colorbar_label = 'T_e (eV)';
        figure_title = 'SOLPS Electron Temperature';
    case {'ti', 'ti_j'}
        colorbar_label = 'T_i (J)';
        figure_title = 'SOLPS Ion Temperature';
    case 'ti_ev'
        colorbar_label = 'T_i (eV)';
        figure_title = 'SOLPS Ion Temperature';
    case 'po'
        colorbar_label = 'p_o';
        figure_title = 'SOLPS Pressure';
    case 'li_tot'
        colorbar_label = 'n_{Li,tot} (m^{-3})';
        figure_title = 'SOLPS Total Lithium Density';
    case 'li_frac'
        colorbar_label = 'n_{Li,tot} / n_e';
        figure_title = 'SOLPS Lithium Fraction';
    otherwise
        error('Unsupported solps_quantity: %s', char(quantity_mode));
end

if should_use_log_plot(quantity_mode, plot_mode)
    colorbar_label = sprintf('log_{10}(%s)', colorbar_label);
end
end


function panel_title = get_panel_title(run_path)
panel_title = 'SOLPS-ITER';

run_log_path = fullfile(run_path, 'run.log');
if ~isfile(run_log_path)
    panel_title = get_panel_title_from_b2mn(run_path, panel_title);
    return;
end

run_log_txt = fileread(run_log_path);
tokens = regexp(run_log_txt, ...
    'stopping because itim >= ntim:\s*(\d+)\s+(\d+)', ...
    'tokens');

if isempty(tokens)
    panel_title = get_panel_title_from_b2mn(run_path, panel_title);
    return;
end

last_tokens = tokens{end};
panel_title = format_solps_iter_title(last_tokens{1});
end


function name = get_last_path_component(path_str)
[~, name, ext] = fileparts(char(path_str));
name = [name, ext];
end


function panel_title = get_panel_title_from_b2mn(run_path, default_title)
panel_title = default_title;

b2mn_path = fullfile(run_path, 'b2mn.dat');
if ~isfile(b2mn_path)
    return;
end

b2mn_txt = fileread(b2mn_path);
tokens = regexp(b2mn_txt, ...
    '''b2mndr_ntim''\s+''(\d+)''', ...
    'tokens', 'once');

if isempty(tokens)
    return;
end

panel_title = format_solps_iter_title(tokens{1});
end


function panel_title = format_solps_iter_title(step_token)
step_count = str2double(step_token);
if ~isfinite(step_count)
    panel_title = sprintf('SOLPS-ITER - %s steps', step_token);
    return;
end

panel_title = sprintf('SOLPS-ITER - %.0f steps', step_count);
end


function panel = load_openedge_panel_data(dump_path, mass_loss_path, wall_path, dt_oe, color_mode)
[R, Z, ts, ids, ~, pmass] = read_case_dump_full(dump_path);
[seg_r, seg_z, seg_values, colorbar_label] = build_openedge_segments( ...
    R, Z, ts, ids, pmass, mass_loss_path, dt_oe, color_mode);
[wall_r, wall_z] = read_surface_outline(wall_path);

panel = struct( ...
    'seg_r', seg_r, ...
    'seg_z', seg_z, ...
    'seg_values', seg_values, ...
    'wall_r', wall_r, ...
    'wall_z', wall_z, ...
    'colorbar_label', colorbar_label);
end


function [seg_r, seg_z, seg_values, colorbar_label] = build_openedge_segments( ...
    R, Z, ts, ids, pmass, mass_loss_path, dt_oe, color_mode)

switch lower(char(color_mode))
    case 'mass_loss_pct'
        sample_values = compute_mass_loss_percent_per_row(ids, ts, pmass);
        colorbar_label = 'Droplet mass loss (%)';
    case 'cum_li_atoms'
        sample_values = map_timesteps_to_cumulative_atoms(ts, mass_loss_path, dt_oe);
        colorbar_label = 'Cumulative evaporated Li (atoms)';
    otherwise
        error('Unsupported openedge_color_mode: %s', char(color_mode));
end

[~, ord] = sortrows([ids(:), ts(:)], [1 2]);
ids_s = ids(ord);
R_s = R(ord);
Z_s = Z(ord);
val_s = sample_values(ord);

u = unique(ids_s);
seg_r_cell = cell(numel(u), 1);
seg_z_cell = cell(numel(u), 1);
seg_v_cell = cell(numel(u), 1);
n_keep = 0;

for i = 1:numel(u)
    sel = find(ids_s == u(i));
    if numel(sel) < 2
        continue;
    end

    r0 = R_s(sel(1:end-1));
    r1 = R_s(sel(2:end));
    z0 = Z_s(sel(1:end-1));
    z1 = Z_s(sel(2:end));
    v0 = val_s(sel(1:end-1));
    v1 = val_s(sel(2:end));

    vseg = nan(size(v0));
    both = isfinite(v0) & isfinite(v1);
    vseg(both) = 0.5 * (v0(both) + v1(both));
    only0 = isfinite(v0) & ~isfinite(v1);
    vseg(only0) = v0(only0);
    only1 = ~isfinite(v0) & isfinite(v1);
    vseg(only1) = v1(only1);

    good = isfinite(r0) & isfinite(r1) & isfinite(z0) & isfinite(z1) & isfinite(vseg);
    if ~any(good)
        continue;
    end

    n_keep = n_keep + 1;
    seg_r_cell{n_keep} = [r0(good), r1(good)];
    seg_z_cell{n_keep} = [z0(good), z1(good)];
    seg_v_cell{n_keep} = vseg(good);
end

if n_keep == 0
    error('No valid OpenEdge trajectory segments were found for plotting.');
end

seg_r = vertcat(seg_r_cell{1:n_keep});
seg_z = vertcat(seg_z_cell{1:n_keep});
seg_values = vertcat(seg_v_cell{1:n_keep});
end


function sample_values = map_timesteps_to_cumulative_atoms(ts, mass_loss_path, dt_oe)
LI_ATOM_MASS_KG = 6.94 * 1.66053906660e-27;

ml_blocks = read_mass_loss_dump(mass_loss_path);
n_blocks = numel(ml_blocks);
ml_ts = zeros(n_blocks, 1);
atoms_step = zeros(n_blocks, 1);

for ib = 1:n_blocks
    ml_ts(ib) = ml_blocks(ib).timestep;
    [~, ~, atoms_step_blk, ~] = extract_atoms_per_step_from_mass_loss_block( ...
        ml_blocks(ib), LI_ATOM_MASS_KG);
    atoms_step(ib) = sum(atoms_step_blk);
end

[ml_ts, ord] = sort(ml_ts);
atoms_step = atoms_step(ord);
cum_atoms = cumsum(atoms_step);

sample_values = interp1(double(ml_ts), cum_atoms, double(ts), 'previous', 'extrap');
sample_values(ts < ml_ts(1)) = 0.0;

if ~isempty(dt_oe) && isfinite(dt_oe) && (dt_oe > 0)
    sample_values = max(sample_values, 0.0);
end
end


function [r, z] = read_surface_outline(surface_path)
raw = splitlines(fileread(surface_path));
num_points = NaN;
points_idx = [];

for i = 1:numel(raw)
    line_i = strtrim(raw{i});
    if isempty(line_i)
        continue;
    end
    if isnan(num_points)
        tok = regexp(line_i, '^(\d+)\s+points$', 'tokens', 'once');
        if ~isempty(tok)
            num_points = str2double(tok{1});
        end
    end
    if strcmpi(line_i, 'Points')
        points_idx = i;
        break;
    end
end

if ~isfinite(num_points) || isempty(points_idx)
    error('Could not parse points section in %s', surface_path);
end

coords = nan(num_points, 2);
n_found = 0;
for i = (points_idx + 1):numel(raw)
    vals = sscanf(raw{i}, '%f');
    if numel(vals) < 3
        continue;
    end
    n_found = n_found + 1;
    coords(n_found, :) = vals(2:3).';
    if n_found == num_points
        break;
    end
end

if n_found ~= num_points
    error('Expected %d points in %s, found %d.', num_points, surface_path, n_found);
end

r = coords(:, 1);
z = coords(:, 2);
if abs(r(1) - r(end)) > 0 || abs(z(1) - z(end)) > 0
    r(end + 1, 1) = r(1);
    z(end + 1, 1) = z(1);
end
end


function cbar = plot_openedge_panel(ax, OpenEdge, SepOverlay, x_limits, y_limits)
hold(ax, 'on');
box(ax, 'on');

X = [OpenEdge.seg_r, nan(size(OpenEdge.seg_r, 1), 1)];
Y = [OpenEdge.seg_z, nan(size(OpenEdge.seg_z, 1), 1)];
Z = zeros(size(X));
C = [OpenEdge.seg_values, OpenEdge.seg_values, nan(size(OpenEdge.seg_values))];

surface(ax, X, Y, Z, C, ...
    'FaceColor', 'none', ...
    'EdgeColor', 'interp', ...
    'LineWidth', 1.8, ...
    'MeshStyle', 'row');

plot(ax, OpenEdge.wall_r, OpenEdge.wall_z, 'k-', 'LineWidth', 2.8);
if ~isempty(SepOverlay)
    plot(ax, SepOverlay.r, SepOverlay.z, 'Color', [0.85, 0.15, 0.15], 'LineWidth', 3.0);
    if isfield(SepOverlay, 'r2') && ~isempty(SepOverlay.r2)
        plot(ax, SepOverlay.r2, SepOverlay.z2, 'Color', [0.85, 0.15, 0.15], 'LineWidth', 3.0);
    end
end

axis(ax, 'equal');
% xlim(ax, x_limits);
% ylim(ax, y_limits);
view(ax, 2);
grid(ax, 'on');
set(ax, 'Layer', 'top');
colormap(ax, turbo);
xlabel(ax, 'R (m)', 'fontsize', 16, 'FontWeight', 'bold');
ylabel(ax, 'Z (m)', 'fontsize', 16, 'FontWeight', 'bold');
title(ax, 'Droplet Trajectories', 'FontSize', 18, 'FontWeight', 'bold');

finite_vals = OpenEdge.seg_values(isfinite(OpenEdge.seg_values));
if ~isempty(finite_vals)
    vmin = min(finite_vals);
    vmax = max(finite_vals);
    if vmax <= vmin
        vmax = vmin + eps(vmin + 1.0);
    end
    caxis(ax, [vmin vmax]);
end

cbar = colorbar(ax);
cbar.Label.String = OpenEdge.colorbar_label;
end


function [ax, cbar] = plot_solps_2d_profile_a(Case, thisData, myTitle, varargin)

if nargin < 3
    myTitle = '';
end

p = inputParser;
p.addParameter('ShowColorbar', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('ShowXLabel', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('ShowYLabel', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('ColorbarLabel', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

plotSep = 1;
plotSep_g = 0;
plotLim_g = 0;
overlayCells = 1;
fontSize = 16;
titleFontSize = 18;
meshLineWidth = 0.45;
wallLineWidth = 2.8;
sepLineWidth = 3.0;
sepColor = [0.85, 0.15, 0.15];

if overlayCells
    edgeCol = 'k';
else
    edgeCol = 'none';
end

ax = gca;
cbar = gobjects(0);
hold(ax, 'on');
box(ax, 'on');
set(ax, 'FontSize', fontSize);

if Case.Geo.isUnstructured
    patch(ax, 'Faces', Case.Geo.Faces, 'Vertices', Case.Geo.Vertices, ...
        'FaceColor', 'flat', 'FaceVertexCData', thisData(:), ...
        'edgecolor', edgeCol, 'linewidth', meshLineWidth, 'edgealpha', 0.05);
else
    if all(size(thisData) == [Case.Geo.nx+2, Case.Geo.ny+2])
        patch(ax, Case.Geo.pr, Case.Geo.pz, thisData(:), ...
            'edgecolor', edgeCol, 'linewidth', meshLineWidth, 'edgealpha', 0.05);
    elseif all(size(thisData) == [Case.Tri.geo.nt, 1])
        patch(ax, Case.Tri.geo.trix.', Case.Tri.geo.triy.', thisData, ...
            'edgecolor', edgeCol, 'linewidth', meshLineWidth, 'edgealpha', 0.05);
    else
        fprintf('\n\nERROR:   ')
        fprintf('                  Data size: [ '); fprintf('%d ', size(thisData)); fprintf(']\n')
        fprintf('Expected either plasma mesh: [ %d %d ]\n', [Case.Geo.nx+2, Case.Geo.ny+2])
        fprintf('           or triangle mesh: [ %d %d ]\n', [Case.Tri.geo.nt, 1])
        error('Did not recognize data dimensions, did you forget a species index?')
    end
end

axis(ax, 'equal');

if opts.ShowColorbar
    cbar = colorbar(ax);
    cbar.FontSize = fontSize;
    if strlength(string(opts.ColorbarLabel)) > 0
        cbar.Label.String = char(opts.ColorbarLabel);
    end
end

if opts.ShowXLabel
    xlabel(ax, 'R [m]', 'fontsize', fontSize, 'FontWeight', 'bold');
else
    xlabel(ax, '');
end

if opts.ShowYLabel
    ylabel(ax, 'Z [m]', 'fontsize', fontSize, 'FontWeight', 'bold');
else
    ylabel(ax, '');
    set(ax, 'YTickLabel', []);
end

title(ax, myTitle, 'FontSize', titleFontSize, 'FontWeight', 'bold');

wall = plot_mesh_extra(Case.run_path, 0, 0);
if ~isempty(wall)
    plot(ax, wall.r, wall.z, 'k-', 'LineWidth', wallLineWidth);
end

if ~Case.Geo.isUnstructured && plotSep
    Sep = get_sep_from_SOLPS_grid(Case.Geo);
    if ~isempty(Sep)
        plot(ax, Sep.r, Sep.z, 'Color', sepColor, 'LineWidth', sepLineWidth);
        if isfield(Sep, 'r2')
            plot(ax, Sep.r2, Sep.z2, 'Color', sepColor, 'LineWidth', sepLineWidth);
        end
    end
end

if plotSep_g && ~isempty(Case.g)
    contour(ax, Case.g.r, Case.g.z, Case.g.psirz.', [1, 1] * Case.g.ssibry, ...
        'Color', sepColor, 'LineWidth', sepLineWidth);
end

if plotLim_g && ~isempty(Case.g)
    plot(ax, Case.g.lim(1, Case.g.lim(1, :) > 0), ...
        Case.g.lim(2, Case.g.lim(1, :) > 0), 'k-', 'LineWidth', wallLineWidth);
end

end


function [run_root, snapshot_dir, snapshot_name] = resolve_case_paths(input_path, state_source_mode, snapshot_subdir)
input_path = char(input_path);
snapshot_name = "";

if isfile(input_path)
    [snapshot_dir, name, ext] = fileparts(input_path);
    snapshot_name = string([name, ext]);
    run_root = fileparts(snapshot_dir);
    return;
end

if ~isfolder(input_path)
    error('Path does not exist: %s', input_path);
end

if strcmpi(char(state_source_mode), 'run_state')
    if strcmp(get_last_path_component(input_path), 'b2mn.exe.dir')
        run_root = fileparts(input_path);
    else
        run_root = input_path;
    end
    snapshot_dir = '';
    return;
end

if strcmp(get_last_path_component(input_path), 'b2mn.exe.dir')
    snapshot_dir = input_path;
    run_root = fileparts(input_path);
else
    run_root = input_path;
    snapshot_dir = fullfile(run_root, char(snapshot_subdir));
end

if ~isfolder(snapshot_dir)
    error('Snapshot directory not found: %s', snapshot_dir);
end
end


function [State, snapshot_label] = load_plasmastate_state(snapshot_dir, snapshot_name, Geo, templateState)
snapshot_file = select_plasmastate_file(snapshot_dir, snapshot_name);
raw = read_plasmastate_subset(snapshot_file, {'na', 'ne', 'te', 'ti', 'po', 'time'});
meta = read_snapshot_state_metadata(fullfile(snapshot_dir, 'b2fstate'));

nxg = Geo.nx + 2;
nyg = Geo.ny + 2;
cell_count = nxg * nyg;
ns = meta.ns;
if (~isfinite(ns) || ns <= 0) && isfield(raw, 'na')
    ns = numel(raw.na) / cell_count;
end

if abs(ns - round(ns)) > 0
    error('Could not infer an integer species count from %s.', snapshot_file);
end
ns = round(ns);

if isstruct(templateState) && ~isempty(templateState)
    State = templateState;
else
    State = struct();
end

State.zamin = meta.zamin;
State.zamax = meta.zamax;
State.zn = meta.zn;
State.am = meta.am;
if isfield(raw, 'ne')
    State.ne = reshape(raw.ne, nxg, nyg);
end
if isfield(raw, 'te')
    State.te = reshape(raw.te, nxg, nyg);
end
if isfield(raw, 'ti')
    State.ti = reshape(raw.ti, nxg, nyg);
end
if isfield(raw, 'po')
    State.po = reshape(raw.po, nxg, nyg);
end
if isfield(raw, 'na')
    State.na = reshape(raw.na, nxg, nyg, ns);
end

snapshot_label = get_last_path_component(snapshot_file);
if isfield(raw, 'time') && isscalar(raw.time)
    snapshot_label = sprintf('%s  t = %.6g s', snapshot_label, raw.time);
end
end


function meta = read_snapshot_state_metadata(file_name)
txt = fileread(file_name);
lines = regexp(txt, '\r?\n', 'split');

meta = struct();
meta.nx = NaN;
meta.ny = NaN;
meta.ns = NaN;
meta.zamin = [];
meta.zamax = [];
meta.zn = [];
meta.am = [];

i = 1;
while i <= numel(lines)
    header = regexp(lines{i}, '^\*cf:\s+(\w+)\s+(\d+)\s+(.+?)\s*$', 'tokens', 'once');
    if isempty(header)
        i = i + 1;
        continue;
    end

    type_name = header{1};
    count = str2double(header{2});
    field_name = strtrim(header{3});

    i = i + 1;
    payload_lines = {};
    while i <= numel(lines) && isempty(regexp(lines{i}, '^\*cf:', 'once'))
        payload_lines{end + 1} = lines{i}; %#ok<AGROW>
        i = i + 1;
    end
    payload_text = strtrim(strjoin(payload_lines, newline));

    switch field_name
        case 'nx,ny,ns'
            vals = sscanf(payload_text, '%f');
            if numel(vals) >= 3
                meta.nx = vals(1);
                meta.ny = vals(2);
                meta.ns = vals(3);
            end
        case {'zamin', 'zamax', 'zn', 'am'}
            if strcmp(type_name, 'real') || strcmp(type_name, 'int')
                vals = sscanf(payload_text, '%f');
                if numel(vals) < count
                    error('Expected %d values for %s in %s, found %d.', count, field_name, file_name, numel(vals));
                end
                meta.(field_name) = vals(1:count).';
            end
    end
end
end


function snapshot_file = select_plasmastate_file(snapshot_dir, snapshot_name)
snapshot_name = string(snapshot_name);

files = dir(fullfile(snapshot_dir, 'plasmastate.*'));
if isempty(files)
    error('No plasmastate.* files found in %s', snapshot_dir);
end

names = sort(string({files.name}));
if strcmpi(snapshot_name, "latest")
    snapshot_file = fullfile(snapshot_dir, char(names(end)));
    return;
end

snapshot_file = fullfile(snapshot_dir, char(snapshot_name));
if ~isfile(snapshot_file)
    error('Requested snapshot file not found: %s', snapshot_file);
end
end


function data = read_plasmastate_subset(file_name, wanted_names)
fid = fopen(file_name, 'r', 'ieee-le');
if fid < 0
    error('Could not open %s', file_name);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

version_bytes = read_fortran_record(fid);
if isempty(version_bytes)
    error('Empty plasmastate file: %s', file_name);
end

data = struct();
while true
    header_bytes = read_fortran_record(fid);
    if isempty(header_bytes)
        break;
    end

    [type_name, count, field_name] = parse_plasmastate_header(header_bytes);
    payload_bytes = read_fortran_record(fid);
    if isempty(payload_bytes)
        error('Missing payload for field %s in %s', field_name, file_name);
    end

    if any(strcmp(field_name, wanted_names))
        data.(field_name) = decode_plasmastate_payload(type_name, count, payload_bytes);
    end
end
end


function [type_name, count, field_name] = parse_plasmastate_header(header_bytes)
if numel(header_bytes) < 20 || ~strcmp(char(header_bytes(1:4).'), '*cf:')
    error('Malformed plasmastate header record encountered.');
end

type_name = strtrim(char(header_bytes(9:16).'));
count = double(typecast(uint8(header_bytes(17:20)), 'uint32'));
field_name = strtrim(char(header_bytes(21:end).'));
end


function values = decode_plasmastate_payload(type_name, count, payload_bytes)
switch type_name
    case 'real'
        values = double(typecast(uint8(payload_bytes), 'double'));
    case 'int'
        values = double(typecast(uint8(payload_bytes), 'int32'));
    case 'char'
        values = strtrim(char(payload_bytes.'));
    otherwise
        error('Unsupported plasmastate type: %s', type_name);
end

if isnumeric(values) && numel(values) ~= count
    error('Field record size mismatch for %s data.', type_name);
end
end


function record_bytes = read_fortran_record(fid)
n_bytes = fread(fid, 1, 'uint32=>double');
if isempty(n_bytes)
    record_bytes = [];
    return;
end

record_bytes = fread(fid, n_bytes, 'uint8=>uint8');
if numel(record_bytes) ~= n_bytes
    error('Short read while parsing Fortran record.');
end

end_bytes = fread(fid, 1, 'uint32=>double');
if isempty(end_bytes) || end_bytes ~= n_bytes
    error('Fortran record markers do not match.');
end
end


function qe = electron_charge()
qe = 1.602176634e-19;
end

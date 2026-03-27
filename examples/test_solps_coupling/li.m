clearvars;
close all;
clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, 'matlab')));
addpath(genpath('/Users/42d/SOLPS-routines/Matlab'));
addpath(genpath('/Users/42d/SOLPS-routines'));

run_paths = {
    % '/Users/42d/solps_iter_oe/v0_300'
    % '/Users/42d/solps_iter_oe/v0_1'
    '/Users/42d/solps_iter_oe/v0_10_steps'
    '/Users/42d/solps_iter_oe/v0_100_steps'
    '/Users/42d/solps_iter_oe/v0_1000_steps'
};

show_openedge_panel = true;
openedge_color_mode = "cum_li_atoms";  % "mass_loss_pct" or "cum_li_atoms"
dump_path = fullfile(script_dir, 'output', 'case.dump');
mass_loss_path = fullfile(script_dir, 'output', 'mass_loss.txt');
wall_path = fullfile(script_dir, 'input', 'wall.surf');
dt_oe = 1.0e-5;

solps_quantity = "li_tot";  % "li_tot" or "li_frac"

x_limits = [2.34 4.0];
y_limits = [-3.8 -2.1];
outdir = fullfile(script_dir, 'output', 'Figs');
fig_base = 'openedge_solps_li_overview';
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
    Cases{i} = load_solps_case(run_paths{i}, 1, "none", "none", "none");
    solps_fields{i} = get_li_field(Cases{i}, solps_quantity);
    panel_titles{i} = get_panel_title(run_paths{i});

    data_i = solps_fields{i};
    finite_i = isfinite(data_i);
    if any(finite_i(:))
        cmin = min(cmin, min(data_i(finite_i)));
        cmax = max(cmax, max(data_i(finite_i)));
    end
end

if ~isfinite(cmin) || ~isfinite(cmax) || (cmax <= cmin)
    cmin = 0.0;
    cmax = 1.0;
end

[colorbar_label, ~] = get_solps_labels(solps_quantity);

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
    xlim(ax(panel_idx), x_limits);
    ylim(ax(panel_idx), y_limits);
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


function li_field = get_li_field(Case, quantity_mode)
iLi = find((Case.State.zn == 3) & (abs(Case.State.am - 7) < 0.2));
Li_tot = sum(Case.State.na(:, :, iLi), 3);

switch lower(char(quantity_mode))
    case 'li_tot'
        li_field = Li_tot;
    case 'li_frac'
        ne = Case.State.ne;
        li_field = zeros(size(Li_tot));
        valid = isfinite(Li_tot) & isfinite(ne) & (ne > 0);
        li_field(valid) = Li_tot(valid) ./ ne(valid);
    otherwise
        error('Unsupported solps_quantity: %s', char(quantity_mode));
end

end


function [colorbar_label, figure_title] = get_solps_labels(quantity_mode)
switch lower(char(quantity_mode))
    case 'li_tot'
        colorbar_label = 'n_{Li,tot} (m^{-3})';
        figure_title = 'SOLPS Total Lithium Density';
    case 'li_frac'
        colorbar_label = 'n_{Li,tot} / n_e';
        figure_title = 'SOLPS Lithium Fraction';
    otherwise
        error('Unsupported solps_quantity: %s', char(quantity_mode));
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
name = string([name, ext]);
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
xlim(ax, x_limits);
ylim(ax, y_limits);
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

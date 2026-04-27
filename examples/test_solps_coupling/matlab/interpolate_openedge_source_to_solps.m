function s_solps_1d = interpolate_openedge_source_to_solps(r_oe, z_oe, s_oe, r_solps, z_solps, vol_solps_1d, total_atoms_s_target)
pts_ok = isfinite(r_oe) & isfinite(z_oe) & isfinite(s_oe);
r_oe = r_oe(pts_ok);
z_oe = z_oe(pts_ok);
s_oe = s_oe(pts_ok);

s_solps_1d = zeros(size(r_solps));
grid_ok = isfinite(r_solps) & isfinite(z_solps) & isfinite(vol_solps_1d) & (vol_solps_1d > 0);
if nnz(pts_ok) < 3 || nnz(grid_ok) == 0
    return;
end

try
    F = scatteredInterpolant(r_oe, z_oe, s_oe, 'linear', 'none');
    r_eval = r_solps(grid_ok);
    z_eval = z_solps(grid_ok);
    s_eval = F(r_eval, z_eval);
catch
    r_eval = r_solps(grid_ok);
    z_eval = z_solps(grid_ok);
    s_eval = griddata(r_oe, z_oe, s_oe, r_eval, z_eval, 'linear');
end

nan_mask = ~isfinite(s_eval);
if any(nan_mask)
    try
        Fnn = scatteredInterpolant(r_oe, z_oe, s_oe, 'nearest', 'nearest');
        s_eval(nan_mask) = Fnn(r_eval(nan_mask), z_eval(nan_mask));
    catch
        s_eval(nan_mask) = griddata(r_oe, z_oe, s_oe, r_eval(nan_mask), z_eval(nan_mask), 'nearest');
    end
end

s_eval(~isfinite(s_eval)) = 0.0;
s_eval = max(s_eval, 0.0);
s_solps_1d(grid_ok) = s_eval;

total_solps_atoms_s = sum(s_solps_1d(grid_ok) .* vol_solps_1d(grid_ok), 'omitnan');
if total_atoms_s_target > 0 && total_solps_atoms_s > 0
    s_solps_1d(grid_ok) = s_solps_1d(grid_ok) * (total_atoms_s_target / total_solps_atoms_s);
end
end

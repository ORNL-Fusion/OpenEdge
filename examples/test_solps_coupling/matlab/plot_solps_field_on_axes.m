function plot_solps_field_on_axes(ax, Geo, data2d, ttl, xlim_plot, ylim_plot, add_colorbar)
% Render SOLPS quadrilateral cells directly on provided axes.
% Geo.pr / Geo.pz are 4 x nCells (ordered vertices).
vals = data2d(:);
patch(ax, Geo.pr, Geo.pz, vals', ...
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
    cb.Label.String = 'Li source (atoms/m^3/s)';
end
colormap(ax, turbo);
if ~all(~isfinite(vals))
    caxis(ax, [min(vals(isfinite(vals))) max(vals(isfinite(vals))) + eps]);
end
end

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

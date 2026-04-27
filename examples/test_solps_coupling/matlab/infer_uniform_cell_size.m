function d = infer_uniform_cell_size(coord)
u = unique(round(coord(:), 12));
if numel(u) < 2
    error('Cannot infer cell size from fewer than 2 unique coordinates.');
end
du = diff(sort(u));
du = du(du > 0.0);
if isempty(du)
    error('Cannot infer positive cell size from coordinates.');
end
d = median(du);
end

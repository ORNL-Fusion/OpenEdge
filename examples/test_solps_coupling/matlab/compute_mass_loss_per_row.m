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

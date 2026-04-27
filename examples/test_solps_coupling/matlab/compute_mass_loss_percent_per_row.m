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

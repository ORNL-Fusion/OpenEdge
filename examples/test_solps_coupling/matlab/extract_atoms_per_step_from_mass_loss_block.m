function [r, z, atoms_step, method] = extract_atoms_per_step_from_mass_loss_block(block, atom_mass_kg)
headers = cellstr(block.headers);
data = block.data;

idx_r = find(strcmp(headers, 'xc'), 1);
idx_z = find(strcmp(headers, 'yc'), 1);
if isempty(idx_r) || isempty(idx_z)
    error('mass_loss block missing xc/yc columns.');
end
r = data(:, idx_r);
z = data(:, idx_z);

dm_idx = find(contains(headers, '[1]') | contains(lower(headers), 'dm'));
dn_idx = find(contains(headers, '[2]') | contains(lower(headers), 'dn'));

if ~isempty(dm_idx)
    dm = sum(data(:, dm_idx), 2, 'omitnan');
    dm = max(dm, 0.0);
    atoms_step = dm / atom_mass_kg;
    method = 'mass_to_atoms';
elseif ~isempty(dn_idx)
    atoms_step = sum(data(:, dn_idx), 2, 'omitnan');
    atoms_step = max(atoms_step, 0.0);
    method = 'atoms_direct';
else
    error('No recognizable Li loss column found in mass_loss block (need [1]/dm or [2]/dn).');
end

atoms_step(~isfinite(atoms_step)) = 0.0;
end

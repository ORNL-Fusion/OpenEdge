function total_atoms_s = write_source2d_file(out_file, atoms_rate_2d, ns, li_idx)
% SOLPS source2d writer:
% For each species block, write ny+2 lines of nx+2 values.
% Li species gets atoms/s per cell, others are zero.
src = max(atoms_rate_2d, 0.0);
[nxg, nyg] = size(src);
zrow = zeros(1, nxg);

fid = fopen(out_file, 'w');
if fid < 0
    error('Could not open source file for writing: %s', out_file);
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>

for is = 1:ns
    for iy = 1:nyg
        if is == li_idx
            row = src(:, iy).';
        else
            row = zrow;
        end
        fprintf(fid, '%.8e', row(1));
        for ix = 2:nxg
            fprintf(fid, ' %.8e', row(ix));
        end
        fprintf(fid, '\n');
    end
end

total_atoms_s = sum(src(:), 'omitnan');
end

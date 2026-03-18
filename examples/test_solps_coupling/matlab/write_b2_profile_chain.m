function write_b2_profile_chain(out_dir, profile_base, src_names, t_switch)
% Write:
%   b2.sources.profile
%   b2.sources.profile.2
%   ...
% with chained sources_filename and sources_time_switch.
n = numel(src_names);
if n < 1
    return;
end

for k = 2:n
    prof_name = sprintf('%s.%d', profile_base, k);
    prof_path = fullfile(out_dir, prof_name);
    fid = fopen(prof_path, 'w');
    if fid < 0
        error('Could not write profile file: %s', prof_path);
    end
    fprintf(fid, '&profile\n');
    fprintf(fid, 'read_sna0_2d=.true.\n');
    fprintf(fid, 'sna0_2d_filename="%s"\n', char(src_names(k)));

    if k < n
        next_prof = sprintf('%s.%d', profile_base, k + 1);
        fprintf(fid, 'sources_time_switch=%s\n', fmt_fortran_d_micro(t_switch(k)));
        fprintf(fid, 'sources_filename="%s"\n', next_prof);
    else
        % Keep final profile valid for SOLPS branches that require
        % sources_time_switch to advance at each profile read.
        fprintf(fid, 'sources_time_switch=1.d9\n');
    end
    fprintf(fid, '/\n');
    fclose(fid);
end

% Also write base file as entry-point alias of .1
base_path = fullfile(out_dir, profile_base);
fid = fopen(base_path, 'w');
if fid < 0
    error('Could not write profile file: %s', base_path);
end
fprintf(fid, '&profile\n');
fprintf(fid, 'read_sna0_2d=.true.\n');
fprintf(fid, 'sna0_2d_filename="%s"\n', char(src_names(1)));
if n > 1
    fprintf(fid, 'sources_time_switch=%s\n', fmt_fortran_d_micro(t_switch(1)));
    fprintf(fid, 'sources_filename="%s.2"\n', profile_base);
end
fprintf(fid, '/\n');
fclose(fid);

fprintf('Wrote %d profile files in %s\n', n, out_dir);
fprintf('Use in b2mn.dat: b2sraI_inputfile=''2'' and profile file %s\n', profile_base);
end

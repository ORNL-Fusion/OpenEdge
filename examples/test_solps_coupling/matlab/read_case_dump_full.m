function [R, Z, ts, ids, types, pmass] = read_case_dump_full(fname)
% Read LAMMPS-like dump and return x->R, y->Z plus id/type/v_pmass.

fid = fopen(fname, 'r');
if fid < 0
    error('Could not open dump file: %s', fname);
end

R = [];
Z = [];
ts = [];
ids = [];
types = [];
pmass = [];

this_ts = NaN;
this_n = NaN;

while true
    tline = fgetl(fid);
    if ~ischar(tline)
        break;
    end
    tline = strtrim(tline);

    if strcmp(tline, 'ITEM: TIMESTEP')
        this_ts = str2double(strtrim(fgetl(fid)));
    elseif strcmp(tline, 'ITEM: NUMBER OF ATOMS')
        this_n = str2double(strtrim(fgetl(fid)));
    elseif startsWith(tline, 'ITEM: BOX BOUNDS')
        fgetl(fid); fgetl(fid); fgetl(fid);
    elseif startsWith(tline, 'ITEM: ATOMS')
        parts = split(tline);
        headers = parts(3:end);
        iid = find(strcmp(headers, 'id'), 1);
        itype = find(strcmp(headers, 'type'), 1);
        ix = find(strcmp(headers, 'x'), 1);
        iy = find(strcmp(headers, 'y'), 1);
        im = find(strcmp(headers, 'v_pmass'), 1);
        if isempty(iid) || isempty(ix) || isempty(iy)
            error('ATOMS fields must include id, x, y.');
        end
        if isempty(itype)
            itype = [];
        end
        if isempty(im)
            im = [];
        end
        if isnan(this_n)
            error('NUMBER OF ATOMS missing before ATOMS block.');
        end

        for k = 1:this_n
            row = strtrim(fgetl(fid));
            vals = split(row);
            ids(end+1,1) = str2double(vals{iid}); %#ok<AGROW>
            if ~isempty(itype)
                types(end+1,1) = str2double(vals{itype}); %#ok<AGROW>
            else
                types(end+1,1) = 1; %#ok<AGROW>
            end
            R(end+1,1) = str2double(vals{ix}); %#ok<AGROW>
            Z(end+1,1) = str2double(vals{iy}); %#ok<AGROW>
            if ~isempty(im)
                pmass(end+1,1) = str2double(vals{im}); %#ok<AGROW>
            else
                pmass(end+1,1) = NaN; %#ok<AGROW>
            end
            ts(end+1,1) = this_ts; %#ok<AGROW>
        end
    end
end

fclose(fid);
end

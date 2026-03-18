function blocks = read_mass_loss_dump(fname)
% Read OpenEdge/SPARTA grid dump written by dump grid / ave/grid.

fid = fopen(fname, 'r');
if fid < 0
    error('Could not open mass_loss dump file: %s', fname);
end

blocks = struct('timestep', {}, 'headers', {}, 'data', {});
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
    elseif strcmp(tline, 'ITEM: NUMBER OF CELLS')
        this_n = str2double(strtrim(fgetl(fid)));
    elseif startsWith(tline, 'ITEM: BOX BOUNDS')
        fgetl(fid); fgetl(fid); fgetl(fid);
    elseif startsWith(tline, 'ITEM: CELLS')
        parts = split(tline);
        headers = parts(3:end);
        if isnan(this_ts) || isnan(this_n)
            error('Incomplete mass_loss block before CELLS in %s', fname);
        end
        data = nan(this_n, numel(headers));
        nrow = 0;
        for k = 1:this_n
            row = fgetl(fid);
            if ~ischar(row)
                break;
            end
            vals = split(strtrim(row));
            if numel(vals) < numel(headers)
                continue;
            end
            nrow = nrow + 1;
            for j = 1:numel(headers)
                data(nrow, j) = str2double(vals{j});
            end
        end
        blocks(end+1).timestep = this_ts; %#ok<AGROW>
        blocks(end).headers = headers;
        blocks(end).data = data(1:nrow, :);
    end
end

fclose(fid);

if isempty(blocks)
    error('No timestep blocks found in mass_loss dump: %s', fname);
end
end

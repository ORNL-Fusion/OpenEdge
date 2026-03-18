function [r, z, psi] = parse_equilibrium_ascii(equ_file)
% Reads jm/km, r(1:jm), z(1:km), and ((psi(j,k)-psib,...)) blocks.
r = [];
z = [];
psi = [];
jm = [];
km = [];

fid = fopen(equ_file, 'r');
if fid < 0
    return;
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>

read_r = false;
read_z = false;
read_psi = false;

while true
    tline = fgetl(fid);
    if ~ischar(tline)
        break;
    end
    tok = strsplit(strtrim(tline));
    if isempty(tok)
        continue;
    end

    if numel(tok) >= 3 && strcmp(tok{1}, 'jm') && strcmp(tok{2}, '=')
        jm = str2double(tok{3});
        continue;
    end
    if numel(tok) >= 3 && strcmp(tok{1}, 'km') && strcmp(tok{2}, '=')
        km = str2double(tok{3});
        continue;
    end

    if strcmp(tok{1}, '((psi(j,k)-psib,j=1,jm),k=1,km)')
        read_psi = true; read_z = false; read_r = false;
        continue;
    end
    if strcmp(tok{1}, 'z(1:km);')
        read_z = true; read_r = false; read_psi = false;
        continue;
    end
    if strcmp(tok{1}, 'r(1:jm);')
        read_r = true; read_z = false; read_psi = false;
        continue;
    end

    nums = sscanf(tline, '%f').';
    if isempty(nums)
        read_r = false; read_z = false; read_psi = false;
        continue;
    end
    if read_psi
        psi = [psi, nums]; %#ok<AGROW>
    elseif read_z
        z = [z, nums]; %#ok<AGROW>
    elseif read_r
        r = [r, nums]; %#ok<AGROW>
    end
end

if isempty(jm), jm = numel(r); end
if isempty(km), km = numel(z); end
if isempty(r) || isempty(z) || isempty(psi)
    r = []; z = []; psi = [];
    return;
end

jm = min(jm, numel(r));
km = min(km, numel(z));
r = r(1:jm);
z = z(1:km);
nneed = jm * km;
if numel(psi) < nneed
    r = []; z = []; psi = [];
    return;
end
psi = reshape(psi(1:nneed), [jm, km]).';
end

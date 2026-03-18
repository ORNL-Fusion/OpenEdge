function [rCore, zCore] = get_core_from_equ_file(equ_file, psi_core_n, rSep, zSep)
% Parse .equ text file directly (same style as Python readEquilibrium),
% then extract one clean psi_N core contour.
rCore = [];
zCore = [];
if ~isfile(equ_file)
    return;
end

[r, z, psi_delta] = parse_equilibrium_ascii(equ_file);
if isempty(r) || isempty(z) || isempty(psi_delta)
    return;
end

% For files containing (psi - psib): separatrix is near 0, axis is min.
psi_axis = min(psi_delta(:), [], 'omitnan');
psi_sep = 0.0;
if nargin >= 4 && ~isempty(rSep) && ~isempty(zSep)
    [RR, ZZ] = meshgrid(r, z);
    psi_sep_s = interp2(RR, ZZ, psi_delta, rSep(:), zSep(:), 'linear', nan);
    if any(isfinite(psi_sep_s))
        psi_sep = median(psi_sep_s(isfinite(psi_sep_s)));
    end
end
den = psi_sep - psi_axis;
if ~isfinite(den) || abs(den) <= eps
    return;
end

psiN = (psi_delta - psi_axis) ./ den;
C = contourc(r, z, psiN, [psi_core_n psi_core_n]);
[rCore, zCore] = contourc_longest_segment(C);
end

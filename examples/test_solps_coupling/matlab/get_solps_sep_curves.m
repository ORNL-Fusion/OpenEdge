function [rSep, zSep, rSep2, zSep2] = get_solps_sep_curves(Geo)
% Use SOLPS-native topology-aware separatrix curves.
rSep = [];
zSep = [];
rSep2 = [];
zSep2 = [];

if ~isfield(Geo, 'Sep') || isempty(Geo.Sep) || ~isfield(Geo,'jsep')
    return;
end
if isfield(Geo.Sep, 'r') && isfield(Geo.Sep, 'z')
    rSep = Geo.Sep.r(:);
    zSep = Geo.Sep.z(:);
end
if isfield(Geo.Sep, 'r2') && isfield(Geo.Sep, 'z2')
    rSep2 = Geo.Sep.r2(:);
    zSep2 = Geo.Sep.z2(:);
end
end

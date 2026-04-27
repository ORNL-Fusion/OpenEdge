function s = fmt_fortran_d(x)
% Return Fortran D-format scalar string (e.g. 1.000000d-04).
s = lower(sprintf('%.6e', x));
s = strrep(s, 'e', 'd');
end

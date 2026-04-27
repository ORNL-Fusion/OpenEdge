function s = fmt_fortran_d_micro(x)
% Prefer microsecond-style Fortran formatting used in SOLPS examples,
% e.g. 100.d-6, 200.d-6, ...
us = x / 1e-6;
if abs(us - round(us)) < 1e-9
    s = sprintf('%.0f.d-6', round(us));
else
    s = fmt_fortran_d(x);
end
end

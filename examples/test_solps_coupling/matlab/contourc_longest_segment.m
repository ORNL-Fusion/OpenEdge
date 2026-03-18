function [xseg, yseg] = contourc_longest_segment(C)
% Keep only the longest contour segment to avoid tiny dot-like artifacts.
xseg = [];
yseg = [];
if isempty(C)
    return;
end
k = 1;
best_len = -inf;
best_x = [];
best_y = [];
while k < size(C,2)
    n = C(2,k);
    if n <= 1
        break;
    end
    idx = (k+1):(k+n);
    if idx(end) > size(C,2)
        break;
    end
    x = C(1,idx).';
    y = C(2,idx).';
    seglen = sum(hypot(diff(x), diff(y)), 'omitnan');
    if seglen > best_len
        best_len = seglen;
        best_x = x;
        best_y = y;
    end
    k = k + n + 1;
end
if ~isempty(best_x)
    xseg = best_x;
    yseg = best_y;
end
end

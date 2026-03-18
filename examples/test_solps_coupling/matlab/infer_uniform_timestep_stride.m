function stride = infer_uniform_timestep_stride(ts)
tsu = unique(ts(:));
if numel(tsu) < 2
    stride = 1;
    return;
end
dts = diff(tsu);
dts = dts(dts > 0);
if isempty(dts)
    stride = 1;
else
    stride = max(1, round(median(dts)));
end
end

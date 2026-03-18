function dt_block_s = infer_dump_block_dt_seconds(ts, dt_oe)
tsu = ts(:);
n = numel(tsu);
if n == 0
    dt_block_s = zeros(0, 1);
    return;
end
stride = infer_uniform_timestep_stride(tsu);
dt_block_s = zeros(n, 1);
if n == 1
    dt_block_s(1) = stride * dt_oe;
    return;
end
for i = 1:n-1
    dt_block_s(i) = max((tsu(i+1) - tsu(i)) * dt_oe, eps);
end
dt_block_s(n) = max(stride * dt_oe, eps);
end

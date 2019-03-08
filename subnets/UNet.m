function U = UNet(u, f, b)

L = length(f);

pad = size(f{1}.value, 1);
pad = (pad-1) / 2;
u = vl_nnconv(u, f{1}, b{1}, 'pad', pad);
u = vl_nnrelu(u);
u = vl_nnpool(u, [2,2], 'stride', 2);
u = Upsample(u);

for l=2:L-1
    pad = size(f{l}.value, 1);
    pad = (pad-1) / 2;
    u = vl_nnnormalize(u, [5,1,2e-5,0.75]);
    u = vl_nnconv(u, f{l}, b{l}, 'pad', pad);
    u = vl_nnrelu(u);
    u = vl_nnpool(u, [2,2], 'stride', 2);
    u = Upsample(u);
end

pad = size(f{L}.value, 1);
pad = (pad-1) / 2;
u = vl_nnconv(u, f{L}, b{L}, 'pad', pad);
U = vl_nnrelu(u);

end

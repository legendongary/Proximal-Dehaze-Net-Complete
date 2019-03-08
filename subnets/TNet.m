function T = TNet(t, f, b)

L = length(f);

pad = size(f{1}.value, 1);
pad = (pad-1) / 2;
t = vl_nnconv(t, f{1}, b{1}, 'pad', pad);
t = vl_nnrelu(t);
t = vl_nnpool(t, [2,2], 'stride', 2);
t = Upsample(t);

for l=2:L-1
    pad = size(f{l}.value, 1);
    pad = (pad-1) / 2;
    t = vl_nnnormalize(t, [5,1,2e-5,0.75]);
    t = vl_nnconv(t, f{l}, b{l}, 'pad', pad);
    t = vl_nnrelu(t);
    t = vl_nnpool(t, [2,2], 'stride', 2);
    t = Upsample(t);
end

pad = size(f{L}.value, 1);
pad = (pad-1) / 2;
t = vl_nnconv(t, f{L}, b{L}, 'pad', pad);
T = vl_nnsigmoid(t);

end

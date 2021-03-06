function [resim, restt] = ours_tipres_s3_eval(noisy)

stage = 3;
wsize = 7;

% pre-process image
[SX, SY, ~] = size(noisy);
RX = rem(SX, 2);
RY = rem(SY, 2);
noisy = padarray(noisy, [RX, RY], 'post', 'symmetric');
noisy = im2single(noisy);
noisy = gpuArray(noisy);
light = airLight(noisy);

% load network
pnet = load('./models/net-res-s3');
net = Net(pnet.net);
net.move('gpu');

% init variables
[sx, sy, ~] = size(noisy);
I = noisy;
for c=1:3
    I(:,:,c) = noisy(:,:,c) / light(c);
end
J = I;
T = gpuArray.ones(sx, sy, 'single');

ws = floor(min(sx, sy) / 6);
if rem(ws,2)==0
    ws = ws + 1;
end
fs = (ws-1) / 2;
fg = gpuArray.ones(ws, ws, 'single') / ws^2;
ep = 5e-3;

for st = 1:stage
    alpha = net.getValue(sprintf('alpha_%d', st));
    beta  = net.getValue(sprintf('beta_%d', st));
    gamma = net.getValue(sprintf('gamma_%d', st));
    ae = exp(alpha);
    be = exp(beta);
    ge = exp(gamma);
    [fu, bu] = getParam(net, st, 'U');
    [ft, bt] = getParam(net, st, 'T');
    U = dy_nnupdU(I, J, T, be, ge, wsize);
    U = cat(3, U, I);
    U = UNET(U, fu, bu);
    T = dy_nnupdT(I, J, U, ae, be, wsize);
    T = cat(3, T, I);
    T = TNET(T, ft, bt);
    T = GNET(T, I, fg, fs, ep);
    J = dy_nnupdJ(I, T);
end

for c=1:3
    J(:,:,c) = J(:,:,c) * light(c);
end

% post-processing image
resim = gather(J);
resim = double(resim);
resim = max(min(resim, 1), 0);
restt = double(gather(T));
restt = max(min(restt, 1), 0);

resim = resim(1:end-RX, 1:end-RY, :);
restt = restt(1:end-RX, 1:end-RY, :);

end

function light = airLight(image)

image = gpuArray(image);
darkc = dy_dark_channel(image, 15);
[~, index] = sort(gather(darkc(:)), 'descend');
index = index(floor(0.001*numel(darkc)));

light = zeros(3,1);
image = reshape(gather(image), [], 3);
for c=1:3
    tpimg = image(:, c);
    light(c) = tpimg(index);
end

end

function [f, b] = getParam(net, st, type)

f = cell(4,1);
b = cell(4,1);
for l=1:4
    f{l} = net.getValue(['filt' upper(type) sprintf('_%d%d',st,l)]);
    b{l} = net.getValue(['bias' upper(type) sprintf('_%d%d',st,l)]);
end

end

function U = UNET(u, f, b)

L = length(f);

pad = size(f{1}, 1);
pad = (pad-1) / 2;
u = vl_nnconv(u, f{1}, b{1}, 'pad', pad);
u = vl_nnrelu(u);
u = vl_nnpool(u, [2,2], 'stride', 2);
u = dy_nnupsample(u);

for l=2:L-1
    pad = size(f{l}, 1);
    pad = (pad-1) / 2;
    u = vl_nnconv(u, f{l}, b{l}, 'pad', pad);
    u = vl_nnrelu(u);
    u = vl_nnpool(u, [2,2], 'stride', 2);
    u = dy_nnupsample(u);
end

pad = size(f{L}, 1);
pad = (pad-1) / 2;
u = vl_nnconv(u, f{L}, b{L}, 'pad', pad);
U = vl_nnrelu(u);

end

function T = TNET(t, f, b)

L = length(f);

pad = size(f{1}, 1);
pad = (pad-1) / 2;
t = vl_nnconv(t, f{1}, b{1}, 'pad', pad);
t = vl_nnrelu(t);
t = vl_nnpool(t, [2,2], 'stride', 2);
t = dy_nnupsample(t);

for l=2:L-1
    pad = size(f{l}, 1);
    pad = (pad-1) / 2;
    t = vl_nnconv(t, f{l}, b{l}, 'pad', pad);
    t = vl_nnrelu(t);
    t = vl_nnpool(t, [2,2], 'stride', 2);
    t = dy_nnupsample(t);
end

pad = size(f{L}, 1);
pad = (pad-1) / 2;
t = vl_nnconv(t, f{L}, b{L}, 'pad', pad);
T = vl_nnsigmoid(t);

end

function Q = GNET(P, I, f, w, eps)

I = mean(I, 3);

x0 = dy_nnpad(I, w);
y0 = dy_nnpad(P, w);

x1 = filter2(f, x0, 'valid');
y1 = filter2(f, y0, 'valid');

x2 = filter2(f, x0.*x0, 'valid');
y2 = filter2(f, x0.*y0, 'valid');

x3 = x2 - x1.*x1;
y3 = y2 - x1.*y1;

x4 = y3 ./ (x3+eps);
y4 = y1 - x4.*x1;

clear x0 y0 x1 y1 x2 y2 x3 y3

x5 = dy_nnpad(x4, w);
y5 = dy_nnpad(y4, w);

x6 = filter2(f, x5, 'valid');
y6 = filter2(f, y5, 'valid');

clear x4 y4 x5 y5

Q = x6.*I + y6;

end

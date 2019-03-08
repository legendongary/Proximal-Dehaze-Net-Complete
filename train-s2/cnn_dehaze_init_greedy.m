function net = cnn_dehaze_init_greedy()

noisy = Input();
clean = Input();
ttmap = Input();
darkc = Input();

J0 = GetJ(noisy);
I0 = GetJ(noisy);
T0 = GetT(noisy);

ws = 41;
fs = (ws-1) / 2;
ep = 1e-2;
fg = Param('value', ones(ws,ws,'single')/ws^2, 'learningRate', 0);

net_s1 = load('../train-s1/output/net-epoch-40.mat');
net_s1 = Net(net_s1.net);

alpha_1 = Param('value', net_s1.getValue('alpha_1'), 'learningRate', 0);
alpha_2 = Param('value', net_s1.getValue('alpha_1'), 'learningRate', 1e-4);
beta_1  = Param('value', net_s1.getValue('beta_1'),  'learningRate', 0);
beta_2  = Param('value', net_s1.getValue('beta_1'),  'learningRate', 1e-4);
gamma_1 = Param('value', net_s1.getValue('gamma_1'), 'learningRate', 0);
gamma_2 = Param('value', net_s1.getValue('gamma_1'), 'learningRate', 1e-4);

wsize = 7;

ae_1 = exp(alpha_1);
be_1 = exp(beta_1);
ge_1 = exp(gamma_1);
ae_2 = exp(alpha_2);
be_2 = exp(beta_2);
ge_2 = exp(gamma_2);

ft1 = cell(4, 1);
bt1 = cell(4, 1);
fu1 = cell(4, 1);
bu1 = cell(4, 1);

for n=1:4
    ft1{n} = Param('value', net_s1.getValue(sprintf('filtT_1%d', n)), 'name', sprintf('filtT_1%d', n), 'learningRate', 0);
    bt1{n} = Param('value', net_s1.getValue(sprintf('biasT_1%d', n)), 'name', sprintf('biasT_1%d', n), 'learningRate', 0);
    fu1{n} = Param('value', net_s1.getValue(sprintf('filtU_1%d', n)), 'name', sprintf('filtU_1%d', n), 'learningRate', 0);
    bu1{n} = Param('value', net_s1.getValue(sprintf('biasU_1%d', n)), 'name', sprintf('biasU_1%d', n), 'learningRate', 0);
end

ft2 = cell(4, 1);
bt2 = cell(4, 1);
fu2 = cell(4, 1);
bu2 = cell(4, 1);

for n=1:4
    ft2{n} = Param('value', net_s1.getValue(sprintf('filtT_1%d', n)), 'name', sprintf('filtT_2%d', n), 'learningRate', 1);
    bt2{n} = Param('value', net_s1.getValue(sprintf('biasT_1%d', n)), 'name', sprintf('biasT_2%d', n), 'learningRate', 1);
    fu2{n} = Param('value', net_s1.getValue(sprintf('filtU_1%d', n)), 'name', sprintf('filtU_2%d', n), 'learningRate', 1);
    bu2{n} = Param('value', net_s1.getValue(sprintf('biasU_1%d', n)), 'name', sprintf('biasU_2%d', n), 'learningRate', 1);
end

u1 = UpdateU(I0, J0, T0, be_1, ge_1, wsize);
c1 = cat(3, u1, I0);
U1 = UNet(c1, fu1, bu1);
t1 = UpdateT(I0, J0, U1, ae_1, be_1, wsize);
d1 = cat(3, t1, I0);
S1 = TNet(d1, ft1, bt1);
T1 = GNet(S1, I0, fg, fs, ep);
J1 = UpdateJ(I0, T1);

u2 = UpdateU(I0, J1, T1, be_2, ge_2, wsize);
c2 = cat(3, u2, I0);
U2 = UNet(c2, fu2, bu2);
t2 = UpdateT(I0, J1, U2, ae_2, be_2, wsize);
d2 = cat(3, t2, I0);
S2 = TNet(d2, ft2, bt2);
T2 = GNet(S2, I0, fg, fs, ep);
J2 = UpdateJ(I0, T2);

lossJ = Loss(J2, clean, 1);
lossT = Loss(T2, ttmap, 1);
lossU = Loss(U2, darkc, 1);

Layer.workspaceNames();
net = Net(lossJ, lossT, lossU);

end



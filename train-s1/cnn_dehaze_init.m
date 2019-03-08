function net = cnn_dehaze_init()

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

alpha_1 = Param('value', log(1), 'learningRate', 1e-4);
beta_1  = Param('value', log(5), 'learningRate', 1e-4);
gamma_1 = Param('value', log(1), 'learningRate', 1e-4);

wsize = 7;

ae_1 = exp(alpha_1);
be_1 = exp(beta_1);
ge_1 = exp(gamma_1);


[ft1, bt1] = params('t', 1);
[fu1, bu1] = params('u', 1);

u1 = UpdateU(I0, J0, T0, be_1, ge_1, wsize);
c1 = cat(3, u1, I0);
U1 = UNet(c1, fu1, bu1);
t1 = UpdateT(I0, J0, U1, ae_1, be_1, wsize);
d1 = cat(3, t1, I0);
S1 = TNet(d1, ft1, bt1);
T1 = GNet(S1, I0, fg, fs, ep);
J1 = UpdateJ(I0, T1);

lossJ = Loss(J1, clean, 1);
lossT = Loss(T1, ttmap, 1);
lossU = Loss(U1, darkc, 1);

Layer.workspaceNames();
net = Net(lossJ, lossT, lossU);

end



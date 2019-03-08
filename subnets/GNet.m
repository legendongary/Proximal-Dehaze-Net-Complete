function Q = GNet(P, I, f, w, eps)

I = mean(I, 3);

x0 = SymPad(I, w);
y0 = SymPad(P, w);

x1 = vl_nnconv(x0, f, []);
y1 = vl_nnconv(y0, f, []);

x2 = vl_nnconv(x0.*x0, f, []);
y2 = vl_nnconv(x0.*y0, f, []);

x3 = x2 - x1.*x1;
y3 = y2 - x1.*y1;

x4 = y3 ./ (x3+eps);
y4 = y1 - x4.*x1;

x5 = SymPad(x4, w);
y5 = SymPad(y4, w);

x6 = vl_nnconv(x5, f, []);
y6 = vl_nnconv(y5, f, []);

Q = x6.*I + y6;

end


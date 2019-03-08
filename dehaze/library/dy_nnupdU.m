function varargout = dy_nnupdU(I, J, T, beta, gamma, wsize, DZ)

if nargin == 6
    
    U = forward(I, J, T, beta, gamma, wsize);
    varargout = {U};
    
elseif nargin == 7
    
    [DI, DJ, DT, Db, Dg, Dw] = backward(I, J, T, beta, gamma, wsize, DZ);
    varargout = {DI, DJ, DT, Db, Dg, Dw};
    
else
    
    error('Invalid input number.');
    
end

end

function U = forward(I, J, T, beta, gamma, wsize)

ID = dy_dark_channel(I, wsize);
JD = dy_dark_channel(J, wsize);

X1 = T .* (ID + T - 1);
X2 = JD;
Y1 = T .* T;
Y2 = 1;
XX = beta * X1 + gamma * X2;
YY = beta * Y1 + gamma * Y2;

U = XX ./ YY;

end

function [DI, DJ, DT, Db, Dg, Dw] = backward(I, J, T, beta, gamma, wsize, DZ)

[ID, I_index] = dy_dark_channel(I, wsize); %#ok<ASGLU>
[JD, J_index] = dy_dark_channel(J, wsize);

X1 = T .* (ID + T - 1);
X2 = JD;
Y1 = T .* T;
Y2 = 1;
XX = beta * X1 + gamma * X2;
YY = beta * Y1 + gamma * Y2;

% DZDI
DI = 0;

% DZDJ
DJ = DZ .* gamma ./ YY;
DJ = dy_place_back(DJ, J_index, wsize);

% DZDT
DX = beta * (2 * T + ID - 1);
DY = beta * (2 * T);
DT = DX ./ YY - XX .* DY ./ YY ./ YY;
DT = DZ .* DT;

% DZDb
Db = X1 ./ YY - XX .* Y1 ./ YY ./ YY;
Db = DZ .* Db;
Db = sum(Db(:));

% DZDg
Dg = X2 ./ YY - XX .* Y2 ./ YY ./ YY;
Dg = DZ .* Dg;
Dg = sum(Dg(:));

% DZDw
Dw = 0;

end
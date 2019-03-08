function varargout = dy_nnupdT(I, J, U, alpha, beta, wsize, DZ)

if nargin == 6
    
    T = forward(I, J, U, alpha, beta, wsize);
    varargout = {T};
    
elseif nargin == 7
    
    [DI, DJ, DU, Da, Db, Dw] = backward(I, J, U, alpha, beta, wsize, DZ);
    varargout = {DI, DJ, DU, Da, Db, Dw};
    
else
    
    error('Invalid output number.');
    
end

end

function T = forward(I, J, U, alpha, beta, wsize)

ID = dy_dark_channel(I, wsize);

X1 = (J - 1) .* (I - 1);
X1 = sum(X1, 3);
X2 = (U - 1) .* (ID - 1);
Y1 = (J - 1) .* (J - 1);
Y1 = sum(Y1, 3);
Y2 = (U - 1) .* (U - 1);

XX = alpha * X1 + beta * X2;
YY = alpha * Y1 + beta * Y2 + 1e-6;

T = XX ./ YY;

end

function [DI, DJ, DU, Da, Db, Dw] = backward(I, J, U, alpha, beta, wsize, DZ)

ID = dy_dark_channel(I, wsize);

X1 = (J - 1) .* (I - 1);
X1 = sum(X1, 3);
X2 = (U - 1) .* (ID - 1);
Y1 = (J - 1) .* (J - 1);
Y1 = sum(Y1, 3);
Y2 = (U - 1) .* (U - 1);

XX = alpha * X1 + beta * X2;
YY = alpha * Y1 + beta * Y2 + 1e-6;

% DZDI
DI = 0;

% DZDJ
DX = alpha * (I - 1);
DY = alpha * (J - 1) * 2;
DJ = DX ./ YY - XX .* DY ./ YY ./ YY;
DJ = DZ .* DJ;

% DZDU
DX = beta * (ID - 1);
DY = beta * (U - 1) * 2;
DU = DX ./ YY - XX .* DY ./ YY ./ YY;
DU = DZ .* DU;

% DZDa
Da = X1 ./ YY - XX .* Y1 ./ YY ./ YY;
Da = DZ .* Da;
Da = sum(Da(:));

% DZDb
Db = X2 ./ YY - XX .* Y2 ./ YY ./ YY;
Db = DZ .* Db;
Db = sum(Db(:));

% DZDw
Dw = 0;

end
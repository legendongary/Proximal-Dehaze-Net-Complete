function varargout = dy_nnupdJ(I, T, Dz)

if nargin == 2
    
    J = forward(I, T);
    varargout = {J};
    
elseif nargin == 3
    
    [DI, DT] = backward(I, T, Dz);
    varargout = {DI, DT};
    
else
    
    error('Invalid output number.');
    
end

end

function J = forward(I, T)

T = max(T, 0.1);
J = (I - 1) ./ T + 1;

end


function [DI, DT] = backward(I, T, Dz)

T = max(T, 0.1);

DI = 0;

DT = (1 - I) ./ T ./ T;
DT = Dz .* DT;
DT = sum(DT, 3);
DT = DT .* (T >= 0.1);

end
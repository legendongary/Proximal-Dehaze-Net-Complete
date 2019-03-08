function varargout = dy_nnloss(x, l, t, d)

switch t
    case 1
        if nargin == 3
            r = x - l;
            y = sum(abs(r(:)));
            varargout = {y};
        elseif nargin == 4
            y = d .* sign((x - l));
            varargout = {y, 0, 0};
        else
            error('Invalid input to loss function.')
        end
    case 2
        if nargin == 3
            r = x - l;
            y = sum(r(:).^2) / 2;
            varargout = {y};
        elseif nargin == 4
            y = d .* (x - l);
            varargout = {y, 0, 0};
        else
            error('Invalid input to loss function.')
        end
        
    otherwise
        error('No such error function.');
end

end


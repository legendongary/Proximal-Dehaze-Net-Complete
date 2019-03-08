
function [filt, bias] = params(t, n)

switch t
    case 't'
        [filt{1}, bias{1}] = xavier([7,7,4,9], [1e-2, 1], print_name('T',n,1));
        [filt{2}, bias{2}] = xavier([7,7,9,9], [1e-2, 1], print_name('T',n,2));
        [filt{3}, bias{3}] = xavier([7,7,9,9], [1e-2, 1], print_name('T',n,3));
        [filt{4}, bias{4}] = xavier([1,1,9,1], [1e-2, 1], print_name('T',n,4));
    case 'u'
        [filt{1}, bias{1}] = xavier([7,7,4,9], [1e-2, 1], print_name('U',n,1));
        [filt{2}, bias{2}] = xavier([7,7,9,9], [1e-2, 1], print_name('U',n,2));
        [filt{3}, bias{3}] = xavier([7,7,9,9], [1e-2, 1], print_name('U',n,3));
        [filt{4}, bias{4}] = xavier([1,1,9,1], [1e-2, 1], print_name('U',n,4));
    otherwise
        error('No such type of params.');
end

end

function [filt, bias] = xavier(shape, lr, name)

h = shape(1);
w = shape(2);
d = shape(3);
n = shape(4);

n1 = h*w*d;
n2 = h*w*n;

sc = sqrt(6/(n1+n2));
f = sc * (2*rand(h,w,d,n,'single')-1);
b = zeros(1,n,'single');

filt = Param('value', f, 'name', name{1}, 'learningRate', lr(1));
bias = Param('value', b, 'name', name{2}, 'learningRate', lr(2));

end

function name = print_name(t, n, m)

fname = sprintf('filt%s_%d%d', t, n, m);
bname = sprintf('bias%s_%d%d', t, n, m);
name = {fname, bname};

end
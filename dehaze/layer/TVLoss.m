function loss = TVLoss(I1, I2, filter)

IX1 = vl_nnconv(I1, filter{1}, []);
IX2 = vl_nnconv(I2, filter{1}, []);
IY1 = vl_nnconv(I1, filter{2}, []);
IY2 = vl_nnconv(I2, filter{2}, []);

loss_x = abs(IX1 - IX2);
loss_y = abs(IY1 - IY2);

loss = loss_x(:) + loss_y(:);

end
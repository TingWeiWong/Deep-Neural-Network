# -*- coding: utf-8 -*-
import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

# y = w2 * w1 * x
y_size, x_size, mid_dim = 100, 100, 1000

# Create random input and output data
x = torch.randn(x_size, 3, device=device, dtype=dtype)
y = torch.randn(y_size, 3, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(mid_dim, x_size, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(y_size, mid_dim, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-9

for t in range(50000):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = w2.clamp(min=0).mm(w1).mm(x)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
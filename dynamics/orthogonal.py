# -*- coding: utf-8 -*-
import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# y = w2 * w1 * x
y_size, x_size, mid_dim = 100, 100, 100

# Create random input and output data
x = torch.randn(x_size, 50, device=device, dtype=dtype)
y = torch.randn(y_size, 50, device=device, dtype=dtype)

# Randomly initialize weights
# w1 = torch.randn(mid_dim, mid_dim, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(mid_dim, mid_dim, device=device, dtype=dtype, requires_grad=True)

# w1, _ = w1.qr()
# w2, _ = w2.qr()
w1 = torch.empty(mid_dim,mid_dim, device=device, dtype=dtype, requires_grad=True)
w2 = torch.empty(mid_dim,mid_dim, device=device, dtype=dtype,requires_grad=True)

torch.nn.init.orthogonal_(w1, gain=1)
torch.nn.init.orthogonal_(w2, gain=1)


init_equivalent_weight = w2.mm(w1)

_ , S_i, _ = torch.svd(init_equivalent_weight, False, False)

max_singular, min_singular = S_i[0], S_i[-1]

max_allowed_learning_rate = 1 / max_singular

print ("Max, min singular = ", max_singular, min_singular)

print ("Max allowed learning rate = ",max_allowed_learning_rate)

learning_rate = 5e-5

# w1 = torch.randn(mid_dim, x_size, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(y_size, mid_dim, device=device, dtype=dtype, requires_grad=True)

loss_list = []
p_difference_list = []
condition_number_list = []

epoch = int(5E4)


for t in range(epoch):
	# Forward pass: compute predicted y using operations on Tensors; these
	# are exactly the same operations we used to compute the forward pass using
	# Tensors, but we do not need to keep references to intermediate values since
	# we are not implementing the backward pass by hand.

	# y_pred = w2.mm(w1.mm(x))
	# y_pred = x.mm(w1).clamp(min=0).mm(w2)
	y_pred = w2.mm(w1).mm(x)


	equivalent_weight = w2.mm(w1)

	_ , S, _ = torch.svd(equivalent_weight, False, False)

	condition_number = S[0] / S[-1]
	condition_number_list.append(condition_number)

	# print ("condition_number = ",condition_number)

	# Compute and print loss using operations on Tensors.
	# Now loss is a Tensor of shape (1,)
	# loss.item() gets the scalar value held in the loss.
	loss = (y_pred - y).pow(2).sum()

	if t % 100 == 99:
		print(t, "loss = ",loss)
		print ("Condition number = ",condition_number)

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
		w1_update = learning_rate * w1.grad
		w1_update_scale = torch.norm(w1_update)
		w2_update = learning_rate * w2.grad
		w2_update_scale = torch.norm(w2_update)
		w1_scale, w2_scale = torch.norm(w1), torch.norm(w2)

		if t % 100 == 99:
			print ("W1 ratio = ",w1_update_scale / w1_scale)
			print ("W2 ratio = ",w2_update_scale / w1_scale)
		w1 -= w1_update
		w2 -= w2_update

		# Manually zero the gradients after updating weights
		w1.grad.zero_()
		w2.grad.zero_()
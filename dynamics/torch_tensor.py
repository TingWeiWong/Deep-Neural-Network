# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch


dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# y = w2 * w1 * x
y_size, x_size, mid_dim = 100, 100, 1000

# Create random input and output data
x = torch.randn(x_size, 1, device=device, dtype=dtype)
y = torch.randn(y_size, 1, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(mid_dim, x_size, device=device, dtype=dtype)
w2 = torch.randn(y_size, mid_dim, device=device, dtype=dtype)

# Input correlation 

# Input correlation 
input_correlation = x.mm(x.t())

input_output_correlation = y.mm(x.t())

learning_rate = 1e-8

loss_list = []
p_difference_list = []


epoch = int(5E4)

for t in range(epoch):
	# Forward pass: compute predicted y
	h = w1.mm(x)
	# h_relu = np.maximum(h, 0)
	y_pred = w2.mm(h)

	# Compute and print loss
	loss = (y_pred - y).pow(2).sum().item()
	print(t, "loss = ",loss)

	# Backprop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0 * (y_pred - y)
	# grad_w2 = h.t().mm(grad_y_pred)
	grad_w2 = grad_y_pred.mm(h.t())
	inter = w2.t().mm(grad_y_pred)
	grad_w1 = inter.mm(x.t())


	subtraction_term = input_output_correlation - (w2.mm(w1.mm(input_correlation)))
	w1_right = learning_rate * w2.t().mm(subtraction_term)
	w2_right = learning_rate * subtraction_term.mm(w1.t())

	# P1 = grad_w1.mm(np.linalg.inv(w1_right))
	# P2 = grad_w2.mm(np.linalg.inv(w2_right))
	p1 = grad_w1[1][1] / w1_right[1][1] / 2
	p2 = grad_w2[1][1] / w2_right[1][1] / 2
	# print ("P1 value = ",p1)
	# print ("P2 value = ",p2)
	p_difference = p1.norm() - p2.norm()
	print ("P difference = ",p_difference)
	p_difference_list.append(p_difference)
	loss_list.append(loss)
	

	# Update weights
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2






x_axis = list(range(epoch))
# plt.plot(x_axis,p_difference_list,label='p_difference')
plt.plot(x_axis,loss_list, label='loss')
# plt.yscale('log')
# plt.savefig('100_depth_weight_mean_without_xavier_linear_activation')
plt.legend(loc='upper left')
plt.yscale('log')
# plt.show()
plt.savefig('./graph/learning_rate_%s_learning_dynamics_difference' % learning_rate)
# plt.savefig()











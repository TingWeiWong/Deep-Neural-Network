# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# y = w2 * w1 * x
y_size, x_size, mid_dim = 100, 100, 100

learning_rate_list = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]

epoch = int(5E5)
x_axis = list(range(epoch))

loss_dict, condition_number_dict, max_allowed_learning_rate_dict, equivalent_weight_dict = {}, {}, {}, {}

max_singular_dict, min_singular_dict = {}, {}


def calculate_condition_and_max_rate(weight_list):
	"""
	This function transforms weight list into:
		1. condition number list 
		2. max learning_rate list with weight squared
	"""
	condition_number_list, max_allowed_learning_rate_list = [], []
	max_singular_list, min_singular_list = [], []

	for equivalent_weight in weight_list:

		weight_squared = equivalent_weight.t().mm(equivalent_weight)

		_ , S, _ = torch.svd(equivalent_weight, False, False)

		_ , S_squared, _ = torch.svd(weight_squared, False, False)

		max_allowed_learning_rate = 1 / S_squared[0]

		max_singular, min_singular = S[0], S[-1]

		condition_number = max_singular / min_singular
		
		condition_number_list.append(condition_number)
		max_allowed_learning_rate_list.append(max_allowed_learning_rate)
		max_singular_list.append(max_singular)
		min_singular_list.append(min_singular)

	return condition_number_list, max_allowed_learning_rate_list, max_singular_list, min_singular_list








for learning_rate in learning_rate_list:
	# Create random input and output data
	x = torch.randn(x_size, 50, device=device, dtype=dtype)
	y = torch.randn(y_size, 50, device=device, dtype=dtype)

	# Randomly initialize weights
	w1 = torch.randn(mid_dim, mid_dim, device=device, dtype=dtype, requires_grad=True)
	w2 = torch.randn(mid_dim, mid_dim, device=device, dtype=dtype, requires_grad=True)

	init_equivalent_weight = (w2).mm(w1)

	init_weight_squared = init_equivalent_weight.t().mm(init_equivalent_weight)

	_ , S_i, _ = torch.svd(init_equivalent_weight, False, False)

	_ , S_squared, _ = torch.svd(init_weight_squared, False, False)


	max_singular, min_singular = S_i[0], S_i[-1]

	max_allowed_learning_rate = 1 / S_squared[0]

	print ("Max, min singular = ", max_singular, min_singular)

	print ("Max allowed learning rate = ",max_allowed_learning_rate)

	# learning_rate = 1.8e-5

	loss_list = []
	equivalent_weight_list = []
	max_allowed_learning_rate_list = []
	condition_number_list = []



	for t in range(epoch):
		# Forward pass: compute predicted y using operations on Tensors; these
		# are exactly the same operations we used to compute the forward pass using
		# Tensors, but we do not need to keep references to intermediate values since
		# we are not implementing the backward pass by hand.

		# y_pred = w2.mm(w1.mm(x))
		# y_pred = x.mm(w1).clamp(min=0).mm(w2)
		y_pred = w2.mm(w1).mm(x)


		equivalent_weight = (w2).mm(w1)

		equivalent_weight_list.append(equivalent_weight.detach())

		# weight_squared = equivalent_weight.t().mm(equivalent_weight)

		# _ , S, _ = torch.svd(equivalent_weight, False, False)

		# _ , S_squared, _ = torch.svd(weight_squared, False, False)

		# max_allowed_learning_rate = 1 / S_squared[0]

		# condition_number = S[0] / S[-1]
		
		# condition_number_list.append(condition_number)
		# max_allowed_learning_rate_list.append(max_allowed_learning_rate)

		# print ("condition_number = ",condition_number)

		# Compute and print loss using operations on Tensors.
		# Now loss is a Tensor of shape (1,)
		# loss.item() gets the scalar value held in the loss.
		loss = (y_pred - y).pow(2).sum()

		loss_list.append(loss.detach())

		# if t % 100 == 99:
		# 	print(t, "loss = ",loss)
		# 	print ("Condition number = ",condition_number)
		# 	print ("Max allowed learning rate = ",max_allowed_learning_rate)

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
			# w1_update_scale = torch.norm(w1_update)
			w2_update = learning_rate * w2.grad
			# w2_update_scale = torch.norm(w2_update)
			# w1_scale, w2_scale = torch.norm(w1), torch.norm(w2)
			# if t % 100 == 99:
			# 	print ("W1 ratio = ",w1_update_scale / w1_scale)
			# 	print ("W2 ratio = ",w2_update_scale / w1_scale)
			w1 -= w1_update
			w2 -= w2_update

			# Manually zero the gradients after updating weights
			w1.grad.zero_()
			w2.grad.zero_()

	loss_dict[learning_rate] = loss_list
	equivalent_weight_dict[learning_rate] = equivalent_weight_list
	# condition_number_dict[learning_rate] = condition_number_list
	# max_allowed_learning_rate_dict[learning_rate] = max_allowed_learning_rate_list


	# plt.plot(x_axis,loss_list, label='loss')
	# # plt.yscale('log')
	# # plt.savefig('100_depth_weight_mean_without_xavier_linear_activation')
	# plt.legend(loc='upper left')
	# plt.yscale('log')
	# # plt.show()
	# plt.savefig('./graph/learning_rate_%s_loss' % learning_rate)

	# plt.close()


	# plt.plot(x_axis, constant_list, label='constant')

	# plt.legend(loc='upper left')
	# plt.yscale('log')
	# # plt.show()
	# plt.savefig('./graph/learning_rate_%s_constant' % learning_rate)
	# plt.close()




for key in loss_dict:
	plt.plot(x_axis,loss_dict[key], label=str(key))

plt.legend(loc='upper left')

plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('The loss through training')

plt.yscale('log')
plt.savefig('./graph/autograd_loss_%s' % epoch)
plt.close()




for key in equivalent_weight_dict:
	condition_number_dict[key], max_allowed_learning_rate_dict[key], max_singular_dict[key], min_singular_dict[key] = calculate_condition_and_max_rate(equivalent_weight_dict[key])



for key in condition_number_dict:
	plt.plot(x_axis,condition_number_dict[key], label=str(key))

plt.legend(loc='upper left')

plt.xlabel('epoch')
plt.ylabel('Condition number')
plt.title('The condition number of W2W1 through training')

plt.yscale('log')
plt.savefig('./graph/autograd_condition_number_%s' % epoch)
plt.close()


for key in max_allowed_learning_rate_dict:
	plt.plot(x_axis,max_allowed_learning_rate_dict[key], label=str(key))

plt.legend(loc='upper left')

plt.xlabel('epoch')
plt.ylabel('Max allowed learning rate')
plt.title('The max allowed learning rate through training')

plt.yscale('log')
plt.savefig('./graph/autograd_max_learning_rate_%s' % epoch)
plt.close()


for key in max_singular_dict:
	plt.plot(x_axis,max_singular_dict[key], label=str(key))

plt.legend(loc='upper left')

plt.xlabel('epoch')
plt.ylabel('Max singular value')
plt.title('The Max singular value of W2W1 through training')

plt.yscale('log')
plt.savefig('./graph/autograd_max_singular_%s' % epoch)
plt.close()


for key in min_singular_dict:
	plt.plot(x_axis,min_singular_dict[key], label=str(key))

plt.legend(loc='upper left')

plt.xlabel('epoch')
plt.ylabel('Min singular value')
plt.title('The Min singular value of W2W1 through training')

plt.yscale('log')
plt.savefig('./graph/autograd_min_singular_%s' % epoch)
plt.close()







# for key in condition_number_dict:
# 	plt.plot(x_axis,condition_number_dict[key], label=str(key))

# plt.legend(loc='upper left')
# plt.yscale('log')
# plt.savefig('./graph/autograd_condition_number')
# plt.close()

# for key in max_allowed_learning_rate_dict:
# 	plt.plot(x_axis,max_allowed_learning_rate_dict[key], label=str(key))

# plt.legend(loc='upper left')
# plt.yscale('log')
# plt.savefig('./graph/autograd_max_learningRate')
# plt.close()



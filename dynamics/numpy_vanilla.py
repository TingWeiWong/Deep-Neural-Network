# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# y = w2 * w1 * x
y_size, x_size, mid_dim = 100, 100, 1000

# Create random input and output data
x = np.random.randn(x_size, 1)
y = np.random.randn(y_size, 1)

# Randomly initialize weights
w1 = np.random.randn(mid_dim, x_size)
w2 = np.random.randn(y_size, mid_dim)

# Input correlation 

# Input correlation 
input_correlation = x.dot(x.T)

input_output_correlation = y.dot(x.T)

learning_rate = 1e-8

p_difference_list = []

for t in range(500):
    # Forward pass: compute predicted y
    h = w1.dot(x)
    # h_relu = np.maximum(h, 0)
    y_pred = w2.dot(h)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, "loss = ",loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    # grad_w2 = h.T.dot(grad_y_pred)
    grad_w2 = grad_y_pred.dot(h.T)
    inter = w2.T.dot(grad_y_pred)
    grad_w1 = inter.dot(x.T)


    subtraction_term = input_output_correlation - (w2.dot(w1.dot(input_correlation)))
    w1_right = learning_rate * w2.T.dot(subtraction_term)
    w2_right = learning_rate * subtraction_term.dot(w1.T)

    # P1 = grad_w1.dot(np.linalg.inv(w1_right))
    # P2 = grad_w2.dot(np.linalg.inv(w2_right))
    p1 = grad_w1[1][1] / w1_right[1][1] / 2
    p2 = grad_w2[1][1] / w2_right[1][1] / 2
    print ("P1 value = ",p1)
    print ("P2 value = ",p2)
    p_difference = np.linalg.norm(p1) - np.linalg.norm(p2)
    print ("P difference = ",p_difference)
    p_difference_list.append(p_difference)
    # ratio = P * learning_rate 
    # print ("ratio = ",ratio)
    P1 = np.divide(grad_w1,w1_right)
    P2 = np.divide(grad_w2,w2_right)

    p1_norm = np.linalg.norm(P1)
    p2_norm = np.linalg.norm(P2)

    difference = p1_norm - p2_norm 

    # print ("difference = ",difference)

    # print ("P1, P2 = ",P1,P2)
    



    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2






x_axis = list(range(500))
plt.plot(x_axis,p_difference_list,label='p_difference')
# plt.yscale('log')
# plt.savefig('100_depth_weight_mean_without_xavier_linear_activation')
plt.legend(loc='upper left')
plt.yscale('log')
# plt.show()
plt.savefig('./graph/learning_rate_%s_learning_dynamics_difference' % learning_rate)












# -*- coding: utf-8 -*-
import numpy as np

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

learning_rate = 1e-5
for t in range(500):
    # Forward pass: compute predicted y
    h = w1.dot(x)
    # h_relu = np.maximum(h, 0)
    y_pred = w2.dot(h)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

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
    print ("P difference = ",np.linalg.norm(p1) - np.linalg.norm(p2))
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
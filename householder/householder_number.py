from random import gauss
import numpy as np 
import matplotlib.pyplot as plt

"""
Since householder matrix H = I - 2vvT for v equals unit vector with the same dimension
of I_n, we want to see if H = H1*H2*H3...Hn if n has any impact on H 
"""



# Global parameters
householder_number = 100
dimension = 100

def gaussian_random_vector(dimension):
	vec = [gauss(0, 1) for i in range(dimension)]
	mag = sum(x**2 for x in vec) ** .5
	# print ("Mag = ",mag)
	unit_vector =  np.transpose([x/mag for x in vec])
	return unit_vector	

def gauss_rand_vector(dims):
	vec = [gauss(0, 1) for i in range(dims)]
	mag = sum(x**2 for x in vec) ** .5
	# print ("Mag = ",mag)
	unit_vector =  np.transpose([x/mag for x in vec])
	unit_vector_transpose = [x/mag for x in vec]
	dot_product = unit_vector.dot(unit_vector_transpose)
	# outer_product = unit_vector_transpose * unit_vector
	outer_product = np.outer(unit_vector_transpose,unit_vector)
	print ("Dot product = ",dot_product)
	outer_product = outer_product / dot_product
	# print("outer_product = ",outer_product)
	# print ("Dot product = ",dot_product)
	# print ("unit_vector_transpose = ",unit_vector_transpose)
	return outer_product

def CreateHouseholder(dimension):
	Identity = np.eye(dimension,dimension)
	householder = Identity - 2 * gauss_rand_vector(dimension)
	# householder_transpose = np.transpose(householder)
	# return householder,householder_transpose
	return householder
householder_matrix = CreateHouseholder(3)
print ("householder = ",np.matmul(householder_matrix,np.transpose(householder_matrix)))

def construct_householder_list(householder_number, dimension):
	householder_list = []
	for i in range(householder_number):
		householder_list.append(CreateHouseholder(dimension))
	for i in range(householder_number):
		I = np.eye(dimension,dimension)
		for j in range(i+1):
			I = np.matmul(I,householder_list[j])
		householder_list[i] = I

	return householder_list

def get_householder_product_list(householder_number,dimension):
	product_list = []
	householder_list = construct_householder_list(householder_number,dimension)

	for matrix in householder_list:
		product_list.append(np.matmul(matrix,np.transpose(matrix)))
	return product_list



def eval_householder_list(householder_number,dimension):
	eval_list = []
	householder_list = construct_householder_list(householder_number,dimension)

	for householder in householder_list:
		eval_list.append(np.linalg.eigvals(householder))
	return eval_list


def calculate_distance_product(householder_number,dimension):
	product_list = get_householder_product_list(householder_number,dimension)
	Identity = np.eye(dimension,dimension)
	dist_list = []
	for product in product_list:
		dist = np.linalg.norm(product-Identity)
		dist_list.append(dist)
	return dist_list


def householder_with_vector_normalization(v_number, dimension):
	error_list = np.zeros((v_number,))

	for h in range(v_number):
		H = np.eye(dimension)
		for _ in range(h):
			v = np.random.normal(size=(dimension, 1))
			H = np.matmul(H, np.eye(dimension) - 2*v@v.T/(v.T@v))
		distance = np.eye(dimension) - np.matmul(H,H.T)
		error_list[h] = np.linalg.norm(distance)

	plt.plot(error_list)
	plt.show()	
	return error_list

error_list = householder_with_vector_normalization(100,100)
# Normalization is what makes householder product explode
# v_number = dimension


# error_list = np.zeros((v_number,))

# for h in range(v_number):
# 	H = np.eye(100)
# 	for _ in range(h):
# 		v = np.random.normal(size=(100, 1))
# 		# H = H@(np.eye(100) - 2*v@v.T/(v.T@v))
# 		H = H@(np.eye(100) - 2*v@v.T)
# 		# H = np.matmul(H, np.eye(100) - 2*v@v.T/(v.T@v))
# 	d = np.eye(100) - H@H.T
# 	error_list[h] = np.linalg.norm(d)

# plt.plot(error_list)
# plt.show()
# householder_list = construct_householder_list(householder_number,dimension)

# product_list = get_householder_product_list(householder_number,dimension)

# print ("Last of product_list = ",product_list)

# # eval_list = eval_householder_list(householder_number,dimension) 

dist_list = calculate_distance_product(householder_number,dimension)

# print ("dist_list = ",dist_list)

# x_axis = list(range(householder_number))
# plt.plot(x_axis,dist_list,label="error")
# plt.legend(loc='upper left')
# plt.yscale('log')
# plt.show()

# householder = CreateHouseholder(4)
# evals = np.linalg.eigvals(householder)
# print ("Evals = ",evals)
# s = plt.hist(evals, bins='auto',density=True)
# plt.show()
# result = np.matmul(householder,householder_transpose)
# print ("Householder product = ",result)



# print ("householder product = ",householder[0].dot(householder[1]))
# print (gauss_rand_vector(3))

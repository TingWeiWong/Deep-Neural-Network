from random import gauss
import numpy as np 
import matplotlib.pyplot as plt

"""
Since householder matrix H = I - 2vvT for v equals unit vector with the same dimension
of I_n, we want to see if H = H1*H2*H3...Hn if n has any impact on H 
"""



# Global parameters
householder_number = 50
dimension = 5

def gauss_rand_vector(dims):
	vec = [gauss(0, 1) for i in range(dims)]
	mag = sum(x**2 for x in vec) ** .5
	# print ("Mag = ",mag)
	unit_vector =  np.transpose([x/mag for x in vec])
	unit_vector_transpose = [x/mag for x in vec]
	dot_product = unit_vector.dot(unit_vector_transpose)
	# outer_product = unit_vector_transpose * unit_vector
	outer_product = np.outer(unit_vector_transpose,unit_vector)
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
# print ("householder = ",householder(3))

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


householder_list = construct_householder_list(householder_number,dimension)

product_list = get_householder_product_list(householder_number,dimension)

print ("Last of product_list = ",product_list[-10])

eval_list = eval_householder_list(householder_number,dimension) 

dist_list = calculate_distance_product(householder_number,dimension)

print ("dist_list = ",dist_list)

x_axis = list(range(householder_number))
plt.plot(x_axis,dist_list,label="error")
plt.legend(loc='upper left')
plt.yscale('log')
plt.show()

# householder = CreateHouseholder(4)
# evals = np.linalg.eigvals(householder)
# print ("Evals = ",evals)
# s = plt.hist(evals, bins='auto',density=True)
# plt.show()
# result = np.matmul(householder,householder_transpose)
# print ("Householder product = ",result)



# print ("householder product = ",householder[0].dot(householder[1]))
# print (gauss_rand_vector(3))

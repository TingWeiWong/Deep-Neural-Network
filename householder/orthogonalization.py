import numpy as np

def modifiedGramSchmidt(A):
	"""
	Gives a orthonormal matrix, using modified Gram Schmidt Procedure
	:param A: a matrix of column vectors
	:return: a matrix of orthonormal column vectors
	"""
	# assuming A is a square matrix
	dim = A.shape[0]
	Q = np.zeros(A.shape, dtype=A.dtype)
	for j in range(0, dim):
		q = A[:,j]
		print (q)
		for i in range(0, j):
			rij = np.vdot(Q[:,i], q)
			q = q - rij*Q[:,i]
			print ("Q after modified = ",q)
		rjj = np.linalg.norm(q, ord=2)
		print ("rij =",rjj)
		if np.isclose(rjj,0.0):
			raise ValueError("invalid input matrix")
		else:
			print ("Q before modified = ",Q[:,j])
			Q[:,j] = q*1.0/rjj
			print ("Q after modified = ",Q[:,j])
	return Q

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
# def normalize(vector):
# 	result = vector/np.linalg.norm(vector)	# return abs(vector)
# 	return result
# print (normalize(np.array([1,2,3])))

def GramSchmidt(matrix):
	"""
	Matrix : 2D array with columns = len(matrix[0]), rows = len(matrix)
	Return : Orthonormal bais matrix
	"""
	columns, rows = len(matrix[0]), len(matrix)

	matrix[:,0] = normalize(matrix[:,0])
	print ("Matrix norm = ",matrix)

	for i in range(1,rows):
		main_column = matrix[:,i]
		print ("Main = ",main_column)
		for j in range(0,i):
			second_column = matrix[:,j]
			print ("second_column = ",second_column)
			length = main_column.dot(second_column)
			main_column -= length * second_column
		matrix[:,i] = normalize(main_column)
		print ("Matrix = ",matrix)
	return matrix


def normalize(v):
    return v / np.sqrt(v.dot(v))

# n = len(A)
# A = np.array([[1,2,3],
# 		 [4,5,6],
# 		 [7,8,9]])
# A[:, 0] = normalize(A[:, 0])

# for i in range(1, n):
#     Ai = A[:, i]
#     for j in range(0, i):
#         Aj = A[:, j]
#         t = Ai.dot(Aj)
#         Ai = Ai - t * Aj
#     A[:, i] = normalize(Ai)

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)
# print (A)
test1 = np.array([[1,2,3],
		 [4,5,6],
		 [7,8,9]])


test2 = np.array([[1,0,2],
		 [4,1,8],
		 [7,2,5]])
print (gram_schmidt(test2))
print (gram_schmidt_columns(test2))
# print (modifiedGramSchmidt(test1))

# print (test1[:,0])
# print (GramSchmidt(test1))






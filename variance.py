import numpy as np
import matplotlib.pyplot as plt


def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dm-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

m1 = rvs()
m2 = rvs()
print ("M1, M2 = ",m1,m2)
# print (m1.dot(m1.T))
print ("M1 dot = ",m1.dot(m1.T))
print ("M2 dot = ",m2.dot(m2.T))
M = np.random.normal(0,1,(4,4))

print('A single matrix', M)
var_zeroth_column = [] # axis = 0
mean_zeroth_column = []
var_zeroth_row = [] # axis = 1 
mean_zeroth_row = [] # axis = 1 
var_all = []
mean_all = []

M = rvs()
for i in range(100):
	# M = nd.dot(M, nd.random.normal(shape=(4,4)))
	# M = np.dot(M, np.random.normal(0,1,(4,4)))
	M = np.dot(M, rvs())
	var_zeroth_column.append(np.var(M, axis=0)[0])
	mean_zeroth_column.append(np.mean(M, axis=0)[0])

	var_zeroth_row.append(np.var(M, axis=1)[0])
	mean_zeroth_row.append(np.mean(M, axis=1)[0])

	var_all.append(np.var(M))
	mean_all.append(np.mean(M))
   

# print('After multiplying 100 matrices', M)
# print ("Row variance = ",len(var_zeroth_row))
# print ("Column variance = ",var_zeroth_column)
# print ("All variance = ",var_all)
test = [[1,2],
		[3,4]]
print ("Column = ",np.mean(test,axis=0))



x_axis = list(range(100))
plt.plot(x_axis,var_zeroth_row,label='Row_variance')
# plt.plot(x_axis,mean_zeroth_row,label='Row_mean') 

# plt.plot(x_axis,var_zeroth_column,label='Column_variance')
# plt.plot(x_axis,mean_zeroth_column,label='Column_mean')


# plt.plot(x_axis,var_all,label='All_variance')
# plt.plot(x_axis,mean_all,label='All_mean')

plt.legend(loc='upper left')
plt.yscale('log')
plt.show()
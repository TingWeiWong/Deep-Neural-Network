import numpy as np
import matplotlib.pyplot as plt

N, M = 1000, 500
Q = N / M
W = np.random.normal(0,1,size=(M,N))
# X shape is M x M
X = (1/N)*np.dot(W.T,W)
evals = np.linalg.eigvals(X)
# plt.hist(evals, bins=100,density=True)
# plt.show()


# n = 2
# reps = 10000

# diffs = np.zeros(reps)
# for r in range(reps):
#     A = np.random.normal(scale=n**-0.5, size=(n,n)) 
#     M = 0.5*(A + A.T)
#     w = np.linalg.eigvalsh(M)
#     diffs[r] = abs(w[1] - w[0])
# plt.hist(diffs, bins=int(reps**0.5))
# plt.legend(loc='upper left')
# plt.title("Eigenvalue distribution")
# plt.show()


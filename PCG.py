import numpy as np
import math

def incomplete_cholesky_factorization(A):
    """ Return the incomplete factorization of A"""
    M = np.copy(A)
    n = A.shape[0]

    for k in range(n):
        print "cholesky", k
        M[k, k] = math.sqrt(M[k, k])
        for i in range((k+1),n):
            if (M[i, k] != 0):
                M[i, k] = M[i, k] / M[k, k]

        for j in range((k+1),n):
            for i in range(j,n):
                if (M[i, j] != 0):
                    M[i, j] = M[i, j] - M[i, k] * M[j, k]


    for i in range(n):
        for j in range(i+1,n):
            M[i, j] = 0.
    return M

def PCG(A, b, x0, tolerance):
    """ solve the equation Ax = b using preconditioned conjugate gradient"""
    M = np.diag(np.diag(A))#incomplete_cholesky_factorization(A) # preconditioning matrix
    x = np.copy(x0)
    r = np.subtract(b, A.dot(x))
    z = np.linalg.inv(M).dot(r)
    p = np.copy(z)
    w = A.dot(p)
    alpha = r.T.dot(z)/(p.T.dot(w))
    x = x + alpha*p
    oldr = np.copy(r)
    oldz = np.copy(z)
    r = r - alpha*w
    k = 1
    while np.linalg.norm(r) > tolerance:
        print "PCG", k
        print np.linalg.norm(r)
        z= np.linalg.inv(M).dot(r)
        beta = r.T.dot(z)/(oldr.T.dot(oldz))
        p = z + beta*p
        w = A.dot(p)
        alpha = r.T.dot(z)/(p.T.dot(w))
        x = x + alpha*p
        oldr = np.copy(r)
        oldz = np.copy(z)
        r= r -alpha*w
        k = k+ 1
    return x

# test
"""A = np.asarray([[4.,1.], [1,3]])
x0 = np.zeros([A.shape[0], 1])
b = np.asarray([[1], [2]])
toler = 10.**(-3)
print PCG(A, b, x0, toler)"""

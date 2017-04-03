import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy as sp

from . import base_class as bc

def haar(n, m):
    '''
    Return a (n * m) random orthogonal matrix following haar distribution.

    argument: integer n, m; (n >= m)
    value: a numpy array with dimension n * m
    algorithm:
        1. generate a (n * m) Gaussian Ensemble
        2. QR decomposition, guarantee diagonal of R positive
        3. Pick Q.
        Reference: "The efficient generation of random orthogonal matrices with an application to condition estimators"
    '''

    if n < m:
        raise ValueError("n must be as large as m.")

    A = npr.normal(0, 1, (n, m))
    Q, R = npl.qr(A, mode = 'reduced')[ : 2]

    for i in xrange(m):
        if R[i, i] < 0:
            Q[:, i] = -Q[:, i]

    return Q


def lowRank(n, m, r, dist, **kwargs):
    '''
    Return a (n * m) low rank matrix
    
    argument: integer n, m, r; (n >= m; r <= min(n, m))
    value: a numpy array with dimension (n * m)
    description:
        we construct the low rank matrix A based on
            A = \sum_{i=1} ^ r lambda_i u_i v_i^T
        where U, Lambda, V are independent with each other
        U, V are haar orthogonal matrix with dimension (n * r), (m * r)
        Lambda is (r * r) with diagonal i.i.d. dist
    '''

    if (n < m) or (r > min(n, m)):
        raise ValueError("(n, m, r) should satisfy n >= m and r <= min(n, m).")

    U = haar(n, r)
    V = haar(m, r)

    if type(dist) == str:
        sampler = getattr(npr, dist)
        L = np.diag(sampler(size = r, **kwargs))
    elif type(dist) == bc.ddist:
        L = np.diag(dist.sample(r))
    else:
        raise TypeError("dist must be either a string specifying the distribution, or of ddist type.")

    return np.dot(U, np.dot(L, V.T))




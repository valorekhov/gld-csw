from libc.math cimport pow
import numpy as np
cimport numpy as np

from q_csw cimport compute_alpha, compute_beta, S

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def probability(np.ndarray[DTYPE_t, ndim=1] q, double med = 0., double iqr = 1., double chi = 0., double xi = 0.6, maxit = 1000L):
	alpha = compute_alpha(xi)
	beta = compute_beta(chi)
	a_plus_b_m1 = alpha + beta -1 
	a_minus_b_m1 = alpha - beta -1
	dv = S(.75, chi, xi) - S(.25, chi, xi)
	ret = [(pow(x, a_plus_b_m1) + pow(1 - x, a_minus_b_m1)) * iqr / dv for x in q]
	return np.array(ret, dtype=DTYPE)

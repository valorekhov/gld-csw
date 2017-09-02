from libc.math cimport pow
import numpy as np
cimport numpy as np

from q_csw cimport compute_alpha, compute_beta, S

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def quantile_density(np.ndarray[DTYPE_t, ndim=1] p, iqr, double chi, double xi):
	alpha = compute_alpha(xi)
	beta = compute_beta(chi)
	a_plus_b_m1 = alpha + beta -1 
	a_minus_b_m1 = alpha - beta -1
	dv = S(.75, chi, xi) - S(.25, chi, xi)
	ret = [(pow(x, a_plus_b_m1) + pow(1 - x, a_minus_b_m1)) * iqr / dv for x in p]
	return np.array(ret, dtype=DTYPE)

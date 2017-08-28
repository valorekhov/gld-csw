from libc.math cimport sqrt, log, pow
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double compute_alpha(double xi):
	return 0.5 * (0.5 - xi) / sqrt(xi * (1. - xi))

cdef double compute_beta(double chi):
	return 0.5 * chi / sqrt(1. - chi * chi)

def S(np.ndarray[DTYPE_t, ndim=1] u, double chi, double xi):
	if chi == 0 and xi == 0.5:
		return [log(x) - log(1. - u) for x in u] #np.log(u) - np.log(1. - u)
	
	if chi != 0 and xi == 0.5 * (1 + chi):
		alpha = compute_alpha(xi)
		return [log(x) - (pow(1. - x, alpha) - 1.) / alpha for x in u]
	
	if chi != 0 and xi == 0.5 * (1. - chi):
		beta = compute_beta(chi)
		return [(pow(x, beta) - 1.) / beta - log(1. - x) for x in u]
	
	alpha = compute_alpha(xi)
	beta = compute_beta(chi)
	a_plus_b = alpha + beta
	a_minus_b = alpha - beta
	a_plus_b_inv = 1 / a_plus_b
	a_minus_b_inv = 1 / a_minus_b

	return [a_plus_b_inv * (pow(x, a_plus_b) - 1) - a_minus_b_inv * (pow(1. - x, a_minus_b) - 1) for x in u]

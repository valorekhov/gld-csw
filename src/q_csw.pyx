from libc.math cimport sqrt, log, pow
import numpy as np
cimport numpy as np

DTYPE = np.float64

cdef double compute_alpha(double xi):
	return 0.5 * (0.5 - xi) / sqrt(xi * (1. - xi))

cdef double compute_beta(double chi):
	return 0.5 * chi / sqrt(1. - chi * chi)

cdef double S(double u, double chi, double xi):
	if chi == 0 and xi == 0.5:
		return [log(x) - log(1. - x) for x in u] #np.log(u) - np.log(1. - u)
	
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

	return a_plus_b_inv * (pow(u, a_plus_b) - 1) - a_minus_b_inv * (pow(1. - u, a_minus_b) - 1)

def quantile(np.ndarray[DTYPE_t, ndim=1] p, double mu = 0., double sigma = 1., double chi = 0., double xi = 0.6):
	s_h = S(.5, chi, xi) 
	d = sigma / (S(.75, chi, xi) - S(.25, chi, xi))
	return np.array([mu + d * (S(x, chi, xi) - s_h) for x in p], dtype=DTYPE)

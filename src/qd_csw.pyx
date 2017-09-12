from libc.math cimport pow
import numpy as np
cimport numpy as np

from q_csw cimport compute_s, CswArgs

DTYPE = np.float64

# TODO: Add S1-S3 cases
cdef double differentiate_s(double x, CswArgs args):
	return pow(x, args.alpha_plus_beta- 1.) + pow(1. - x, args.alpha_minus_beta - 1.)

def quantile_density(np.ndarray[DTYPE_t, ndim=1] p, double iqr = 1., double chi = 0., double xi = 0.6):
	args = CswArgs(chi, xi)
	a_plus_b_m1 = args.alpha_plus_beta -1
	a_minus_b_m1 = args.alpha_minus_beta -1
	dv = compute_s(.75, args) - compute_s(.25, args)
	ret = [(pow(x, a_plus_b_m1) + pow(1 - x, a_minus_b_m1)) * iqr / dv for x in p]
	return np.array(ret, dtype=DTYPE)

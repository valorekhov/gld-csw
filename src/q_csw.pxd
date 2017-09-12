import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t

cdef class CswArgs:
    cdef double chi
    cdef double xi
    cdef double alpha
    cdef double beta
    cdef double alpha_plus_beta
    cdef double alpha_minus_beta
    cdef double alpha_plus_beta_inv
    cdef double alpha_minus_beta_inv


cdef double compute_alpha(double xi)

cdef double compute_beta(double chi)

cdef double compute_s(double u, CswArgs args)


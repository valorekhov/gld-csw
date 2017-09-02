import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t

cdef double compute_alpha(double xi)

cdef double compute_beta(double chi)

cdef double S(double u, double chi, double xi)
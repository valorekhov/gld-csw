import numpy as np
cimport numpy as np

from q_csw cimport CswArgs

ctypedef np.float64_t DTYPE_t

cdef newton_approx(double y, a, b, double tol, CswArgs args, unsigned long max_it = ?)
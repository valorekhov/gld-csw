import numpy as np
cimport numpy as np

from q_csw cimport CswArgs

ctypedef np.float64_t DTYPE_t

cdef double differentiate_s(double u, CswArgs args)
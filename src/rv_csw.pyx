from libc.float cimport DBL_EPSILON
import numpy as np
cimport numpy as np

from q_csw cimport compute_s, CswArgs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

EPS = 2* DBL_EPSILON
ONEMEPS = 1 - 2 * DBL_EPSILON
#XIUNIF1 = .5 - 1./sqrt(5.)
#XIUNIF2 = .5 - 2./sqrt(17.)


def random_var(unsigned long n, double med = 0., double iqr = 1., double chi = 0., double xi = 0.6):
    args = CswArgs(chi, xi)
    c = compute_s(.75, args) - compute_s(.25, args)
    a = med - iqr * compute_s(.5, args) / c
    b = iqr / c

    return [a + b * compute_s(px, args) for px in np.random.uniform(0, 1, n)]

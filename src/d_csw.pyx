from libc.math cimport pow, fabs, INFINITY, NAN, isnan, log
from libc.float cimport DBL_EPSILON
import numpy as np
cimport numpy as np

from q_csw cimport compute_s, CswArgs
from qd_csw cimport differentiate_s
from p_csw cimport newton_approx

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ONEMEPS = 1 - 2 * DBL_EPSILON

def density(np.ndarray[DTYPE_t, ndim=1] x, double med = 0., double iqr = 1., double chi = 0., double xi = 0.6, maxit = 1000L):
    #{.25, .5, .75}
    args = CswArgs(chi, xi)
    c = compute_s(.75, args) - compute_s(.25, args)
    a = med - iqr * compute_s(.5, args) / c;
    b = iqr / c;
    e = iqr / c;

    n = x.shape[0]
    qs = np.argsort(x)
    qs = np.flip(qs, 0)
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(n, dtype=DTYPE)

    px = 0.
    qmin = a + b * compute_s(px, args)
    px = 1.
    qmax = a + b * compute_s(px, args)

    for i  in range(n):                       
        xx = x[qs[i]]
        # print("xx=%f qmin=%f qmax=%f\n" % (xx, qmin, qmax))
        if isnan(xx):                         
            dx = xx          
        elif xx == qmin:                    
            px = 0.
            dx = 1. / (e * differentiate_s(px, args))
        elif xx == qmax:                    
            px = 1.
            dx = 1. / (e * differentiate_s(px, args))
        elif xx < qmin or xx > qmax:
            dx = 0.
        else:                            
            ''' Bounds check '''          
            if isnan(px) or px > ONEMEPS:              
                px = 1.          
                              
            px, maxiter = newton_approx((xx - a) / b, 0., px, 0., args, maxit)
            if maxiter < 0:
                print("Reached maxit for q=%f" % xx)     
            dx = 1. / (e * differentiate_s(px, args))
        d[qs[i]] = dx

    return d

from libc.math cimport pow, fabs, INFINITY, NAN, isnan, log
from libc.float cimport DBL_EPSILON
import numpy as np
cimport numpy as np

from q_csw cimport compute_s, CswArgs
from qd_csw cimport differentiate_s

DTYPE = np.float64

EPS = 2* DBL_EPSILON
ONEMEPS = 1 - 2 * DBL_EPSILON
#XIUNIF1 = .5 - 1./sqrt(5.)
#XIUNIF2 = .5 - 2./sqrt(17.)

# cdef double compute_dS1(double x):
#    return 1. / x + 1. / (1. - x)

cdef newton_approx(double y, a, b, double tol, CswArgs args, unsigned long max_it = 1000):
    cdef double fa, fb
    cdef double c, fc
    cdef double p, q
    cdef double new_step
    cdef double tol_act
    maxit = max_it

    f = compute_s
    df = differentiate_s

    if isnan(y):
       return y

    fa = f(a, args) - y
    fb = f(b, args) - y

    # print('fa=%f, fb=%f, y=%f' % (fa, fb, y))

    ''' test if we have found a root at an endpoint '''
    if fabs(fa) < 2 * EPS:        
        return a, 0 

    if fabs(fb) < 2 * EPS:        
        return b, 0


    ''' if there is no root in the interval use the largest possible
       interval. If the root still does not exists, NaN '''
    if fa * fb > 0:	
        a = 0.
        b = 1.
        fa = f(a, args) - y
        fb = f(b, args) - y
        # print('fa=%f, fb=%f, fa * fb =%f' % (fa, fb, fa * fb))
        if fa * fb > 0: 
            return NAN, 0


    while maxit>0:
        maxit = maxit-1 
        ''' swap data to have b the closest to the root '''
        if fabs(fa) < fabs(fb):            
            foo = a
            a = b
            b = foo
            fc = fa
            fa = fb
            fb = fc


        ''' bisection step '''
        new_step = .5 * (a - b)

        ''' Newton steps '''
        p = fb
        q = df(b, args)

        ''' keep p positive for Newton step conditions '''
        if p > 0.:
            q = -q
        else:
            p = -p

        if (fabs(q) != INFINITY) and (q != 0.) and p < fabs(new_step * q):
            new_step = p/q

        ''' save last iteration '''
        c = b
        fc = fb
        b += new_step
        fb = f(b, args) - y

        tol_act = 2. * EPS * fabs(b) + tol / 2.
        if fabs(new_step) <= tol_act or fb == 0:
            return b, max_it - maxit

        ''' reduce search interval '''
        if (fb * fc) < 0:
            a = c; fa = fc

    ''' failed '''
    return b, -1

def probability(np.ndarray[DTYPE_t, ndim=1] q, double med = 0., double iqr = 1., double chi = 0., double xi = 0.6, maxit = 1000):
    args = CswArgs(chi, xi)
    # print( "alpha=%f beta=%f" % (alpha, beta))
    #{.25, .5, .75}
    c = compute_s(.75, args) - compute_s(.25, args)
    a = med - iqr * compute_s(.5, args) / c
    b = iqr / c

    qmin =  a - b / args.alpha_plus_beta if args.alpha_plus_beta > 0 else -INFINITY
    qmax =  a + b / args.alpha_minus_beta if args.alpha_minus_beta > 0 else INFINITY

    # print("alpha+beta=%f alpha-beta=%f" % (a_plus_b, a_minus_b));
    # print("c=%f a=%f b=%f qmin=%f qmax=%f\n " % (c,a, b, qmin, qmax))

    n = q.shape[0]
    qs = np.argsort(q)
    qs = np.flip(qs, 0)
    cdef np.ndarray[DTYPE_t, ndim=1] p = np.empty(n, dtype=DTYPE)

    px = 1.
    for i  in range(n):                       
        qx = q[qs[i]]
        # print("qx=%f qmin=%f qmax=%f\n" % (qx, qmin, qmax))
        if isnan(qx):                         
            px = qx          
        elif qx <= qmin:                    
            px = 0.
        elif qx >= qmax:                    
            px = 1.
        else:                            
            ''' Bounds check '''          
            if isnan(px) or px > ONEMEPS:              
                px = 1.          
                              
            px, maxiter = newton_approx((qx - a) / b, 0., px, 0., args, maxit)
            if maxiter < 0:
                print("Reached maxit for q=%f" % qx)     
        p[qs[i]] = px

    return p

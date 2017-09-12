from libc.math cimport sqrt, log, pow
import numpy as np
cimport numpy as np

DTYPE = np.float64

cdef double compute_alpha(double xi):
    return 0.5 * (0.5 - xi) / sqrt(xi * (1. - xi))

cdef double compute_beta(double chi):
    return 0.5 * chi / sqrt(1. - chi * chi)

cdef class CswArgs:

    def __cinit__(self, double chi, double xi):
        self.chi = chi
        self.xi = xi
        self.alpha = compute_alpha(xi)
        self.beta = compute_beta(chi)
        self.alpha_plus_beta = self.alpha + self.beta
        self.alpha_minus_beta = self.alpha - self.beta
        self.alpha_plus_beta_inv = 1 / self.alpha_plus_beta
        self.alpha_minus_beta_inv = 1/ self.alpha_minus_beta


cdef double compute_s(double u, CswArgs args):
    chi = args.chi
    xi = args.xi
    alpha = args.alpha
    beta = args.beta

    if chi == 0 and xi == 0.5:
        return [log(x) - log(1. - x) for x in u]

    if chi != 0 and xi == 0.5 * (1 + chi):
        return [log(x) - (pow(1. - x, alpha) - 1.) / alpha for x in u]

    if chi != 0 and xi == 0.5 * (1. - chi):
        return [(pow(x, beta) - 1.) / beta - log(1. - x) for x in u]

    return args.alpha_plus_beta_inv * (pow(u, args.alpha_plus_beta) - 1) - args.alpha_minus_beta_inv * (pow(1. - u, args.alpha_minus_beta) - 1)

def quantile(np.ndarray[DTYPE_t, ndim=1] p, double mu = 0., double sigma = 1., double chi = 0., double xi = 0.6):
    args = CswArgs(chi, xi)
    s_h = compute_s(.5, args)
    d = sigma / (compute_s(.75, args) - compute_s(.25, args))
    return np.array([mu + d * (compute_s(x, args) - s_h) for x in p], dtype=DTYPE)

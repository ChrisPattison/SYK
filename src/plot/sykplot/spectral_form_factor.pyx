# distutils: language=c++
cimport numpy as cnp
import numpy as np
from cython.view cimport array as cvarray

cdef extern from "spectral_form_factor_impl.hpp" namespace "syk_plot":
    void spectral_form_factor_impl(double*, long long, double*, long long, double*, long long)

cpdef spectral_form_factor(t, a): 
    cdef double [::1] t_arr = t.copy()
    cdef double [::1] a_arr = a.copy()
    cdef double [::1] spectral = np.zeros_like(t)
    spectral_form_factor_impl(&t_arr[0], t_arr.shape[0], &a_arr[0], a_arr.shape[0], &spectral[0], spectral.shape[0])
    return spectral


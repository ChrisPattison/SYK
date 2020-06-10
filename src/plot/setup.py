from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'sykplot',
    packages = ['sykplot'],
    ext_modules = cythonize(
        [Extension('spectral_form_factor', [
            'sykplot/spectral_form_factor.pyx',],
            extra_compile_args = [
                '-std=c++17',
                '-O3', '-march=native', '-mtune=native',
                '-fno-math-errno', '-funroll-loops',
                '-fopenmp', ],
            extra_link_args = ['-lgomp'],
            include_dirs = [numpy.get_include()])],
        compiler_directives={'language_level' : 3, })
)


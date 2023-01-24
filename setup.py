from Cython.Build import cythonize
from distutils.core import setup, Extension
import numpy

module = Extension('mudanca', sources = [
       'src/ashift.c',
       'src/mudancamodule.pyx'
])

setup (name = 'mudanca',
       version = '1.4.0',
       author="James Campbell",
       description = 'An implementation of the shift algorithmn for python ported from Darktable & Nshift',
       ext_modules = cythonize([module], compiler_directives={'language_level' : "3"}),
       include_dirs=[numpy.get_include()]
)
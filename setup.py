from Cython.Build import cythonize
from distutils.core import setup, Extension
import numpy

module = Extension('pyshift', sources = [
       'src/ashift.c',
       'src/pyshiftmodule.pyx'
])

setup (name = 'Pyshift',
       version = '1.1',
       author="James Campbell",
       description = 'An implementation of the shift algorithmn for python ported from Darktable & Nshift',
       ext_modules = cythonize([module], compiler_directives={'language_level' : "3"}),
       include_dirs=[numpy.get_include()],
       python_requires=[
              'numpy',
              'opencv-python'
       ]
)
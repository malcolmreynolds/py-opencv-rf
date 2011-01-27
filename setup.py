from distutils.core import setup, Extension

c_module = Extension('opencvrf',
                    sources = ['opencvrf_module.cpp', 'opencvrf_training_opts.cpp'],
                    include_dirs = ['/usr/local/include/opencv','/usr/local/include' ])



setup(name = 'opencvrf',
      version = '1.0',
      description = 'Random Forest training for Numpy',
      py_modules = ['opencvrf'],
      ext_modules = [c_module])

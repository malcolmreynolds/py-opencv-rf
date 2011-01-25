from distutils.core import setup, Extension

c_module = Extension('_opencvrf',
                    sources = ['opencvrf_module.cpp', 'opencvrf_training_opts.cpp'])



setup(name = 'opencvrf',
      version = '1.0',
      description = 'Random Forest training for Numpy',
      py_modules = ['opencvrf'],
      ext_modules = [c_module])

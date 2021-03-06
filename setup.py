from distutils.core import setup, Extension

c_module = Extension('opencvrf',
                     sources = ['opencvrf_module.cpp', 'opencvrf_training_opts.cpp', 'numpy_opencv_conversion.cpp',
                                'active_forests.cpp'],
                     include_dirs = ['/usr/local/include/opencv','/usr/local/include',
                                     '/epd64/Versions/Current/lib/python2.6/site-packages/numpy/core/include/numpy'],
                     runtime_library_dirs = ['/usr/local/lib/'],
                     libraries = ['opencv_core', 'opencv_contrib', 'opencv_ml'])




setup(name = 'opencvrf',
      version = '1.0',
      description = 'Random Forest training for Numpy',
      py_modules = ['opencvrf'],
      ext_modules = [c_module])

#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#define NO_IMPORT_ARRAY
#include "arrayobject.h"

#include "assert.h"
#include "common.h"
#include "except.h"

extern PyObject *NumpyConversionFailedError;

#define SET_NUMPY_CONVERSION_ERROR(msg) SET_ERROR_STRING(NumpyConversionFailedError, msg)

int parse_numpy_array_to_opencv(PyObject *array, void* address) {
    printf("inside parse_numpy_array_to_opencv\n");
    PY_ASSERT(array != NULL, "obj is null");
    PY_ASSERT(address != NULL, "address is null");
    printf("after asserts\n");
   
    if (!PyArray_Check(array)) {
        SET_NUMPY_CONVERSION_ERROR("array failed PyArray_Check()");
        return FAILURE;
    }
   
    //npy_intp* dimensions = PyArray_DIMS(array);


    return FAILURE;
}

#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#define NO_IMPORT_ARRAY
#include "arrayobject.h"

#include "ml.h"

#include "assert.h"
#include "common.h"
#include "except.h"

extern PyObject *NumpyConversionFailedError;

#define SET_NUMPY_CONVERSION_ERROR(msg) SET_ERROR_STRING(NumpyConversionFailedError, msg)

void print_opencv_matrix(const CvMat* input) {
    float *dataPtr = input->data.fl;
    for (unsigned int r = 0; r < input->rows; r++) {
        for (unsigned int c = 0; c < input->cols; c++) {
            printf("%f ",*dataPtr);
            dataPtr++;
        }
        printf("\n");
    }
}


int parse_numpy_array_to_opencv(PyObject *array, void* address) {
    printf("inside parse_numpy_array_to_opencv\n");
    PY_ASSERT(array != NULL, "obj is null");
    PY_ASSERT(address != NULL, "address is null");
    printf("after asserts\n");
    
    if (!PyArray_Check(array)) {
        SET_NUMPY_CONVERSION_ERROR("array failed PyArray_Check()");
        return FAILURE;
    }
    if (!PyArray_ISFLOAT(array)) {
        SET_NUMPY_CONVERSION_ERROR("array is not of type float");
        return FAILURE;
    }
    
    PyArrayObject *real_array = (PyArrayObject *) array;
    int ndims = real_array->nd;
    
    if (ndims == 2) {
        int nrows = real_array->dimensions[0];
        int ncols = real_array->dimensions[1];
        printf("%dx%d array\n", nrows, ncols);
        
        CvMat* cvMat = cvCreateMat(nrows, ncols, CV_32F);
        printf("done cvCreateMat\n");
        
        for (unsigned int i=0; i < nrows; i++) {
            float* thisRowOut = cvMat->data.fl + ncols*i;
            for (unsigned int j=0; j < ncols; j++) {
                thisRowOut[j] = *(double *)(real_array->data + i*real_array->strides[0] + j*real_array->strides[1]);
            }
        }
        printf("parsed matrix:\n");
        print_opencv_matrix(cvMat);
        CvMat** out = (CvMat **) address;
        *out = cvMat;
        return SUCCESS;
    }
    else if (ndims == 1) {
        int sz = real_array->dimensions[0];
        printf("%d element vector\n", sz);
        
        CvMat* cvMat = cvCreateMat(ndims, 1, CV_32F);
        for (unsigned int i=0; i<sz; i++) {
            cvMat->data.fl[i] = *(double *)(real_array->data + i*real_array->strides[0]);
        }
        printf("parsed matrix:\n");
        print_opencv_matrix(cvMat);
        CvMat** out = (CvMat **) address;
        *out = cvMat;
        return SUCCESS;
    }
    else {
        SET_NUMPY_CONVERSION_ERROR("only support matrices, sorry!");
        return FAILURE;
    }
}
/*
void print_numpy_array_type(PyArrayObject *array) {
    switch(array->descr->type) {
    case NPY_BOOLLTR: printf("bool"); break;
    case NPY_BYTELTR: printf("byte"); break;
    case NPY_UBYTELTR: printf("ubyte"); break;
        
*/

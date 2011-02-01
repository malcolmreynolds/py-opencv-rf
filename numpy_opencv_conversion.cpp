#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#define NO_IMPORT_ARRAY
#include "arrayobject.h"

#include "ml.h"

#include "assert.h"
#include "common.h"
#include "except.h"

extern PyObject *NumpyConversionFailedError;


//#define DEBUG

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


int parse_numpy_array_to_opencv(PyObject *array, void* out_address) {
    PY_ASSERT(array != NULL, "obj is null");
    PY_ASSERT(out_address != NULL, "address is null");
    
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
#ifdef DEBUG
        printf("%dx%d array\n", nrows, ncols);
#endif
        
        CvMat* cvMat = cvCreateMat(nrows, ncols, CV_32F);
#ifdef DEBUG
        printf("done cvCreateMat\n");
#endif
        
        for (unsigned int i=0; i < nrows; i++) {
            float* thisRowOut = cvMat->data.fl + ncols*i;
            for (unsigned int j=0; j < ncols; j++) {
                thisRowOut[j] = *(double *)(real_array->data + i*real_array->strides[0] + j*real_array->strides[1]);
            }
        }
#ifdef DEBUG
        printf("parsed matrix:\n");
        print_opencv_matrix(cvMat);
#endif
        CvMat** out = (CvMat **) out_address;
        *out = cvMat;
        return SUCCESS;
    }
    else if (ndims == 1) {
        int sz = real_array->dimensions[0];
#ifdef DEBUG
        printf("%d element vector\n", sz);
#endif        

        CvMat* cvMat = cvCreateMat(ndims, 1, CV_32F);
        for (unsigned int i=0; i<sz; i++) {
            cvMat->data.fl[i] = *(double *)(real_array->data + i*real_array->strides[0]);
        }
#ifdef DEBUG
        printf("parsed matrix:\n");
        print_opencv_matrix(cvMat);
#endif
        CvMat** out = (CvMat **) out_address;
        *out = cvMat;
        return SUCCESS;
    }
    else {
        SET_NUMPY_CONVERSION_ERROR("only support matrices, sorry!");
        return FAILURE;
    }
}

int parse_opencv_array_to_numpy(CvMat* cvmat, void* out_address) {
    PY_ASSERT(cvmat != NULL, "input is null");
    PY_ASSERT(out_address != NULL, "out address is null");

    PY_ASSERT(cvmat->type == CV_32F, "cvmat matrix is of wrong type");

    int nrows = cvmat->rows;
    int ncols = cvmat->cols;

    npy_intp dims[2] = {nrows, ncols};
    PyArrayObject* result = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    
    for (unsigned int i=0; i < nrows; i++) {
        float *thisRowIn = cvmat->data.fl + ncols*i;
        for (unsigned int j=0; j < ncols; j++) {
            *(double *)(result->data + i*result->strides[0] + j*result->strides[1]) = thisRowIn[j];
        }
    }

    *((PyArrayObject **) out_address) = result;
    return SUCCESS;
}
        
/*
void print_numpy_array_type(PyArrayObject *array) {
    switch(array->descr->type) {
    case NPY_BOOLLTR: printf("bool"); break;
    case NPY_BYTELTR: printf("byte"); break;
    case NPY_UBYTELTR: printf("ubyte"); break;
        
*/

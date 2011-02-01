#ifndef __COMMON_H
#define __COMMON_H

//because NULL is failure for the conversion functions called by PyArg_ParseTuple
const int FAILURE = NULL;
const int SUCCESS = 1;

//macros to get access to numpy access. Note if array is an expression this will break
//in a most horrible way!!!data
#define NP_ARRAY_DB_1D(array, i) (*(double *)(array->data + (i)*array->strides[0]))
#define NP_ARRAY_DB_2D(array, i, j) (*(double *)(array->data + (i)*array->strides[0] + (j)*array->strides[1]))

#endif

#ifndef __NUMPY_OPENCV_CONVERSION_H
#define __NUMPY_OPENCV_CONVERSION_H

void print_opencv_matrix(const CvMat* input);
int parse_numpy_array_to_opencv(PyObject *array, void *address);

#endif

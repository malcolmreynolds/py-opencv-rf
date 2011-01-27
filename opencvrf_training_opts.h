#ifndef __OPENCVRF_TRAINING_OPTS_H
#define __OPENCVRF_TRAINING_OPTS_H

#include "ml.h"
#include "opencvrf_module.h"

CvRTParams* create_default_training_options();
int parse_py_dict_to_training_options(PyObject *options, void* address);

#endif

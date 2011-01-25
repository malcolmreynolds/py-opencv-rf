#ifndef __OPENCVRF_TRAINING_OPTS_H
#define __OPENCVRF_TRAINING_OPTS_H

#include "ml.h"

//maximum depth of each tree
const int DEFAULT_MAX_DEPTH = 25;

//if there are fewer than this many samples in a node, don't bother splitting
const int DEFAULT_MIN_SAMPLE_COUNT = 5;

//one of the "stop splitting" criteria variables - still not sure about this?
const float DEFAULT_REGRESSION_ACCURACY = 0.00001;

//only useful if dealing with missing features
const bool DEFAULT_USE_SURROGATES = false;

//relates to the number of categorical types - preclustering will happen before this
const int DEFAULT_MAX_CATEGORIES = 10;

//priors on the probability of each class
float* DEFAULT_PRIORS = NULL;

//whether to calculate the importance of each variable
const bool DEFAULT_CALC_VAR_IMPORTANCE = true;

//size of the random subset of features to test at each node
const int DEFAULT_NUM_ACTIVE_VARS = 10;

//maximum number of trees
const int DEFAULT_MAX_TREE_COUNT = 200;

//These go into the termination of the training procecure
const float DEFAULT_FOREST_ACCURACY = 0.0001;
const int DEFAULT_TERM_CRITERIA_TYPE = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;

CvRTParams* create_default_training_options();
CvRTParams* parse_py_dict_to_training_options(PyObject *options);


#endif

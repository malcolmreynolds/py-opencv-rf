#include <Python.h>

#include "opencvrf_training_opts.h"

CvRTParams* create_default_training_options() {
    return new CvRTParams(DEFAULT_MAX_DEPTH, MIN_SAMPLE_COUNT, REGRESSION_ACCURACY,
                          DEFAULT_USE_SURROGATES, DEFAULT_MAX_CATEGORIES, DEFAULT_PRIORS,
                          DEFAULT_CALC_VAR_IMPORTANCE, DEFAULT_NUM_ACTIVE_VARS, DEFAULT_MAX_TREE_COUNT,
                          DEFAULT_FOREST_ACCURACY, DEFAULT_TERM_CRITERIA_TYPE);
}

CvRTParams

#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#include "arrayobject.h"

#include "common.h"
#include "except.h"

#include "opencvrf_module.h"
#include "numpy_opencv_conversion.h"

#include "active_forests.h"

PyObject *InvalidTrainingOptionsError;
PyObject *NumpyConversionFailedError;
PyObject *RandomForestError;


#define SET_RF_ERROR(msg) SET_ERROR_STRING(RandomForestError, msg)

static CvMat* regression_var_type(unsigned int num_variables) {
    //plus one here is for the output variable (ie all the input features are ordered
    //and so is the output, therefore we have regression)
    CvMat* var_type = cvCreateMat(num_variables + 1, 1, CV_8U);
    cvSet(var_type, cvScalarAll(CV_VAR_ORDERED));
    return var_type;
}



//opencvrf.train(x, y[, opts])
// x: training data
// y: training labels
// opts: options controlling training
static PyObject *opencvrf_train(PyObject* self, PyObject *args) {
    CvMat *training_data = NULL;
    CvMat *training_labels = NULL;
    CvRTParams *training_options = NULL;
    char verbose = NULL;

    //eventually fix this to use a converter function to convert numpy arrays into opencv. amazing!
    if (!PyArg_ParseTuple(args, "O&O&|O&b", 
                          parse_numpy_array_to_opencv, &training_data, 
                          parse_numpy_array_to_opencv, &training_labels,
                          parse_py_dict_to_training_options, &training_options,
                          &verbose)) {
        //failed parse == user didn't supply the compulsory arguments. Return null, and the exception
        //saved will suffice
        if (training_data) cvReleaseMat(&training_data);
        if (training_labels) cvReleaseMat(&training_labels);
        if (training_options) delete training_options;
        return FAILURE;
    }

    if (verbose) {
        printf("Verbose output specified\n");
    }

    //if the user didn't supply anything for training options, it will still be null - create now
    if (training_options == NULL) {
        printf("no training options supplied, using defaults.\n");
        training_options = create_default_training_options();
    }
    
    unsigned int num_samples, num_variables, num_training_labels;
    num_samples = training_data->rows;
    num_variables = training_data->cols;
    num_training_labels = training_data->rows;

    if (training_labels->cols != 1) {
        cvReleaseMat(&training_data);
        cvReleaseMat(&training_labels);
        delete training_options;
        SET_RF_ERROR("training labels should be a column vector");
        return FAILURE;
    }

    if (verbose) {
        printf("training data converted to opencv format. %d samples, each with %d variables\n",
               num_samples, num_variables);
        printf("training labels converted to opencv format. %d labels present\n", num_training_labels);
    }

    if (training_labels->rows != num_samples) {
        cvReleaseMat(&training_data);
        cvReleaseMat(&training_labels);
        delete training_options;
        SET_RF_ERROR("number of elements in training labels doesn't match number of samples");
        return FAILURE;
    }

    //This is to set the fact that we are dealing with regression.
    CvMat* var_type_indicator = regression_var_type(num_variables);

    time_t start_time, end_time;
    start_time = time(NULL);
    CvRTrees *forest = new CvRTrees();
    forest->train(training_data, CV_ROW_SAMPLE, training_labels, NULL, NULL, var_type_indicator, NULL, *training_options);
    end_time = time(NULL);
    
    if (verbose) {
        printf("training done in %fs\n", difftime(end_time, start_time));
    }

    cvReleaseMat(&var_type_indicator);
    cvReleaseMat(&training_data);
    cvReleaseMat(&training_labels);
    delete training_options;

    set_forest_active(forest);
    return Py_BuildValue("l", forest);
}

static PyObject* opencvrf_load(PyObject* self, PyObject *args) {
    char* filename;
    char verbose = NULL;
    if (!PyArg_ParseTuple(args, "s|b", &filename, &verbose)) {
        return FAILURE;
    }

    if (verbose) {
        printf("Trying to load forest from %s\n", filename);
    }

    CvRTrees *forest = new CvRTrees;
    forest->load(filename);

    if (forest->get_tree_count() == 0) {
        SET_RF_ERROR("Could not load random forest");
        delete forest;
        return FAILURE;
    }

    set_forest_active(forest);
    return Py_BuildValue("l", forest);
}

static PyObject* opencvrf_save(PyObject* self, PyObject *args) {
    CvRTrees* forest;
    char* filename;
    char verbose = NULL;
    if (!PyArg_ParseTuple(args, "ls|b", (long **)&forest, &filename, &verbose)) {
        return FAILURE;
    }

    if (verbose) {
        printf("Trying to save forest to %s\n", filename);
    }

    if (!is_active_forest(forest)) {
        SET_RF_ERROR("Tried to save a forest which isn't active");
        return FAILURE;
    }

    forest->save(filename);

    Py_RETURN_NONE;
}

static PyObject* opencvrf_delete(PyObject* self, PyObject *args) {
    CvRTrees* forest;
    if (!PyArg_ParseTuple(args, "l", (long **)&forest)) {
        return FAILURE;
    }

    if (!is_active_forest(forest)) {
        SET_RF_ERROR("Tried to delete a forest which isn't active");
        return FAILURE;
    }

    forest->clear();
    delete forest;

    Py_RETURN_NONE;
}



    

static PyObject* opencvrf_predict(PyObject* self, PyObject *args) {
    CvRTrees *forest = NULL;
    CvMat* predict_data = NULL;

    if (!PyArg_ParseTuple(args, "lO&",
                          (long **)&forest, parse_numpy_array_to_opencv, &predict_data)) {
        return FAILURE;
    }

    if (!is_active_forest(forest)) {
        cvReleaseMat(&predict_data);
        SET_RF_ERROR("passed in a forest pointer which is not active!");
        return FAILURE;
    }
   
    /*
    if (get_feature_dimensionality(forest) != predict_data->cols) {
        cvReleaseMat(&predict_data);
        SET_RF_ERROR("data for predictiong is of wrong dimensionality.");
        return FAILURE;
        }*/
    
    CvMat sample; //use this to point at each row in turn
    unsigned int num_to_predict = predict_data->rows;

    printf("predicting on %d samples\n", num_to_predict);

    npy_intp dims[] = {num_to_predict, 1};
    PyArrayObject* results = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    for (unsigned int i=0; i<num_to_predict; i++) {
        cvGetRow(predict_data, &sample, i);
        printf("predicting on row:\n");
        print_opencv_matrix(&sample);
        NP_ARRAY_DB_1D(results, i) =  (double)forest->predict_variance(&sample, NULL);
        printf("answer = %f\n", NP_ARRAY_DB_1D(results, i));
    }

    cvReleaseMat(&predict_data);

    Py_INCREF(results);
    return (PyObject *) results;
}

static PyMethodDef OpencvrfMethods[] = {
    //    {"test", opencvrf_test, METH_VARARGS, "Test"},
    {"delete", opencvrf_delete, METH_VARARGS, "Delete a forest"},
    {"save", opencvrf_save, METH_VARARGS, "Save a forest"},
    {"load", opencvrf_load, METH_VARARGS, "Load a forest"},
    {"train", opencvrf_train, METH_VARARGS, "Train a forest"},
    {"predict", opencvrf_predict, METH_VARARGS, "Predict using a forest"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initopencvrf(void) {
    PyObject *m;
    m = Py_InitModule("opencvrf", OpencvrfMethods);
    if (m == NULL) {
        return;
    }

    //std::set<int> myset;

    import_array();

    InvalidTrainingOptionsError = PyErr_NewException("opencvrf.invalid_training_options_error", NULL, NULL);
    Py_INCREF(InvalidTrainingOptionsError);
    PyModule_AddObject(m, "error", InvalidTrainingOptionsError);

    NumpyConversionFailedError = PyErr_NewException("opencvrf.numpy_conversion_failed_error", NULL, NULL);
    Py_INCREF(NumpyConversionFailedError);
    PyModule_AddObject(m, "error", NumpyConversionFailedError);

    RandomForestError = PyErr_NewException("opencvrf.random_forest_error", NULL, NULL);
    Py_INCREF(RandomForestError);
    PyModule_AddObject(m, "error", RandomForestError);
}

#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#include "arrayobject.h"

#include "opencvrf_module.h"
#include "numpy_opencv_conversion.h"



PyObject *InvalidTrainingOptionsError;
PyObject *NumpyConversionFailedError;

/*
static PyObject * opencvrf_test(PyObject* self, PyObject *args) {
    return Py_BuildValue("i", 42);
}
*/

//opencvrf.train(x, y[, opts])
// x: training data
// y: training labels
// opts: options controlling training
static PyObject *opencvrf_train(PyObject* self, PyObject *args) {
    const PyObject *training_data;
    const PyObject *training_labels;
    const CvRTParams *training_options = NULL;

    //eventually fix this to use a converter function to convert numpy arrays into opencv. amazing!
    if (!PyArg_ParseTuple(args, "O&O&|O&", 
                          parse_numpy_array_to_opencv, &training_data, 
                          parse_numpy_array_to_opencv, &training_labels,
                          parse_py_dict_to_training_options, &training_options)) {
        //failed parse == user didn't supply the compulsory arguments. Return null, and the exception
        //saved will suffice
        return NULL;
    }

    //if the user didn't supply anything for training options, it will still be null - create now
    if (training_options == NULL) {
        printf("no training options supplied, using defaults.\n");
        training_options = create_default_training_options();
    }
    /*
    printf("objects supplied:\nx=%p\ny=%p\nopts=%p\n",
           training_data, training_labels, training_options);
    
    printf("training options as number: %d\n", *((int *)training_options));
    */

    printf("training options:\n");
    printf("max_depth = %d\n", training_options->max_depth);
    printf("min_sample_count = %d\n", training_options->min_sample_count);
    Py_RETURN_NONE;
}
    


static PyMethodDef OpencvrfMethods[] = {
    //    {"test", opencvrf_test, METH_VARARGS, "Test"},
    {"train", opencvrf_train, METH_VARARGS, "Train a forest"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initopencvrf(void) {
    PyObject *m;
    m = Py_InitModule("opencvrf", OpencvrfMethods);
    if (m == NULL) {
        return;
    }

    import_array();

    InvalidTrainingOptionsError = PyErr_NewException("opencvrf.invalid_training_options_error", NULL, NULL);
    Py_INCREF(InvalidTrainingOptionsError);
    PyModule_AddObject(m, "error", InvalidTrainingOptionsError);

    NumpyConversionFailedError = PyErr_NewException("opencvrf.numpy_conversion_failed_error", NULL, NULL);
    Py_INCREF(NumpyConversionFailedError);
    PyModule_AddObject(m, "error", NumpyConversionFailedError);
}

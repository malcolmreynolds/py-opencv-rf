#include <Python.h>

static PyObject * opencvrf_test(PyObject* self, PyObject *args) {
    return Py_BuildValue("i", 42);
}


static PyObject *create_default_training_options() {
    return NULL;
}



//opencvrf.train(x, y[, opts])
// x: training data
// y: training labels
// opts: options controlling training
static PyObject *opencvrf_train(PyObject* self, PyObject *args) {
    const PyObject *training_data;
    const PyObject *training_labels;
    const CvRTParams *training_options = NULL;

    //eventually fix this to use a converter function to convert numpy arrays into opencv. amazing!
    if (!PyArg_ParseTuple(args, "OO|O&", &training_data, &training_labels, &training_options)) {
        //failed parse == user didn't supply the compulsory arguments. Return null, and the exception
        //saved will suffice
        return NULL;
    }

    //if the user didn't supply anything for training options, it will still be null - create now
    if (training_options == NULL) {
        printf("no training options supplied, using defaults.\n");
        training_options = create_default_training_options();
    }

    printf("objects supplied:\nx=%x\ny=%x\nopts=%x\n",
           training_data, training_labels, training_options);

    Py_RETURN_NONE;
}
    


static PyMethodDef OpencvrfMethods[] = {
    {"test", opencvrf_test, METH_VARARGS, "Test"},
    {"train", opencvrf_train, METH_VARARGS, "Train a forest"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initopencvrf(void) {
    (void) Py_InitModule("opencvrf", OpencvrfMethods);
}

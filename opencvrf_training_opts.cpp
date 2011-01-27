#include <Python.h>

#include "opencvrf_training_opts.h"
#include "assert.h"

extern PyObject *InvalidTrainingOptionsError;

#define DEBUG

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

CvRTParams* create_default_training_options() {
    return new CvRTParams(DEFAULT_MAX_DEPTH, DEFAULT_MIN_SAMPLE_COUNT, DEFAULT_REGRESSION_ACCURACY,
                          DEFAULT_USE_SURROGATES, DEFAULT_MAX_CATEGORIES, DEFAULT_PRIORS,
                          DEFAULT_CALC_VAR_IMPORTANCE, DEFAULT_NUM_ACTIVE_VARS, DEFAULT_MAX_TREE_COUNT,
                          DEFAULT_FOREST_ACCURACY, DEFAULT_TERM_CRITERIA_TYPE);
}

typedef enum option_type {
    OP_INT,
    OP_FLOAT,
    OP_FLOAT_ARRAY,
    OP_BOOL
};
    

typedef struct dict_option_parse_info {
    char* name;
    option_type type;
    void *var;
};

//because NULL is failutre
const int FAILURE = NULL;
const int SUCCESS = 1;

 
int parse_int_option_from_object(PyObject* obj, int* dest) {
    PY_ASSERT(obj != NULL, "parse_int_option passed a null obj");
    PY_ASSERT(dest != NULL, "parse_int_option passed a null dest");
    *dest = 1;
    return SUCCESS;
}   

int parse_float_option_from_object(PyObject* obj, float* dest) {
    PY_ASSERT(obj != NULL,"parse_float_option passed a null obj");
    PY_ASSERT(dest != NULL,"parse_float_option passed a null dest");
    *dest = 1.0f;
    return SUCCESS;
}

int parse_bool_option_from_object(PyObject* obj, bool* dest) {
    PY_ASSERT(obj != NULL, "parse_bool_option was passed a null obj");
    PY_ASSERT(dest != NULL, "parse_bool_option was passed a null dest");
    *dest = true;
    return SUCCESS;
}
        

int parse_py_dict_to_training_options(PyObject *option_dict, void *address) {
    printf("inside parse_py_dict_to_training_options, inputs are options=%p address=%p\n", option_dict, address);
    if (!PyDict_Check(option_dict)) {
        PyErr_SetString(InvalidTrainingOptionsError, "Object provided for training options is not a dictionary");
        return 0;
    }
    
    //setup the parameters here. we will then check if any of them have been
    //overridden by the user in the dictionary supplied. This is a bit more complex
    //than it *could* have been, but I think more concise at the same time, and much
    //easier to add new parameters.
    int max_depth = DEFAULT_MAX_DEPTH;
    int min_sample_count = DEFAULT_MIN_SAMPLE_COUNT;
    float regression_accuracy = DEFAULT_REGRESSION_ACCURACY;
    bool use_surrogates = DEFAULT_USE_SURROGATES;
    int max_categories = DEFAULT_MAX_CATEGORIES;
    float *priors = DEFAULT_PRIORS;
    bool calc_var_importance = DEFAULT_CALC_VAR_IMPORTANCE;
    int num_active_vars = DEFAULT_NUM_ACTIVE_VARS;
    int max_tree_count = DEFAULT_MAX_TREE_COUNT;
    int forest_accuracy = DEFAULT_FOREST_ACCURACY;
    int term_criteria_type = DEFAULT_TERM_CRITERIA_TYPE;
    
    //all possble options
    dict_option_parse_info all_options[] = {
        {"max_depth", OP_INT, &max_depth},
        {"min_sample_count", OP_INT, &min_sample_count},
        {"regression_accuracy", OP_FLOAT, &regression_accuracy},
        {"use_surrogates", OP_BOOL, &use_surrogates},
        {"max_categories", OP_INT, &max_categories},
        {"priors", OP_FLOAT_ARRAY, &priors},
        {"calc_var_importance", OP_BOOL, &calc_var_importance},
        {"num_active_vars", OP_INT, &num_active_vars},
        {"max_tree_count", OP_INT, &max_tree_count},
        {"forest_accuracy", OP_FLOAT, &forest_accuracy},
        {"term_criteria_type", OP_INT, &term_criteria_type}
    };
    const int num_options = sizeof(all_options) / sizeof(dict_option_parse_info);

    PyObject* choice;
    char* option_name;
    int status;
    //loop through all the options we have, and see if any of them are present 
    //in the dictionary. If so, parse from a python object and set to our C variable.
    //Once we have tried to look for every option, we will construct the CvRTParams object.
    for (int i=0; i < num_options; i++) {
#ifdef DEBUG
        printf("Checking if %s is present in option dict... ", all_options[i].name);
#endif
        choice = PyDict_GetItemString(option_dict, all_options[i].name);
        if (choice != NULL) {
            printf("%s present!\n", all_options[i].name);
            switch(all_options[i].type) {
            case OP_INT:
                status = parse_int_option_from_object(choice, (int *)(all_options[i].var));
                break;
            case OP_FLOAT:
                status = parse_float_option_from_object(choice, (float *)(all_options[i].var));
                break;
            case OP_BOOL:
                status = parse_bool_option_from_object(choice, (bool *)(all_options[i].var));
                break;
            case OP_FLOAT_ARRAY:
                PyErr_SetString(InvalidTrainingOptionsError, "Object provided for float array. Not yet supported!");
                return NULL;
            default:
                PyErr_SetString(InvalidTrainingOptionsError, "Internal Error: Fell through option_type switch statement! Please report as a bug.");
                return NULL;
            }
            
            //check if we got an error parsing anything from the dictionary - if we did then 
            //the exception should already be set, so just return NULL and it should propagate
            //back to the python interpreter
            if (status == NULL) {
                return NULL;
            }
        }
#ifdef DEBUG
        else {
            printf("not present.\n");
        }
#endif
    }
    
    return 1;
}


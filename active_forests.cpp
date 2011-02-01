/*
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#include "arrayobject.h"
*/
#include <map>

#include "active_forests.h"

struct random_forest_data {
    CvRTrees *forest;
    unsigned int feature_dimensionality;
};

static std::map<CvRTrees*, random_forest_data*> active_forests_info;

bool is_active_forest(CvRTrees* forest) {
    std::map<CvRTrees*, random_forest_data*>::iterator it;

    it = active_forests_info.find(forest);
    return (it != active_forests_info.end());
}

void set_forest_active(CvRTrees* forest, unsigned int feature_dimensionality) {
    random_forest_data* forest_data = new random_forest_data();
    forest_data->forest = forest;
    forest_data->feature_dimensionality = feature_dimensionality;
    active_forests_info[forest] = forest_data;
}

void set_forest_inactive(CvRTrees *forest) {
    active_forests_info.erase(forest);
}

unsigned int get_feature_dimensionality(CvRTrees *forest) {
    std::map<CvRTrees*, random_forest_data*>::iterator it;
    it = active_forests_info.find(forest);
    return (*it).second->feature_dimensionality;
}

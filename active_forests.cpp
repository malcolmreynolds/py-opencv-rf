/*
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL pyopencv_ARRAY_API
#include "arrayobject.h"
*/
#include <set>

#include "active_forests.h"

struct random_forest_data {
    CvRTrees *forest;
    unsigned int feature_dimensionality;
};

static std::set<CvRTrees*> active_forests_info;

bool is_active_forest(CvRTrees* forest) {
    std::set<CvRTrees*>::iterator it;

    it = active_forests_info.find(forest);
    return (it != active_forests_info.end());
}

void set_forest_active(CvRTrees* forest) {
    active_forests_info.insert(forest);
}

void set_forest_inactive(CvRTrees *forest) {
    active_forests_info.erase(forest);
}


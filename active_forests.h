#ifndef __ACTIVE_FORESTS_H
#define __ACTIVE_FORESTS_H

#include "ml.h"

bool is_active_forest(CvRTrees* forest);
void set_forest_active(CvRTrees* forest, unsigned int feature_dimensionality);
void set_forest_inactive(CvRTrees* forest);
unsigned int get_feature_dimensionality(CvRTrees* forest);


#endif

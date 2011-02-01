#ifndef __ACTIVE_FORESTS_H
#define __ACTIVE_FORESTS_H

#include "ml.h"

bool is_active_forest(CvRTrees* forest);
void set_forest_active(CvRTrees* forest);
void set_forest_inactive(CvRTrees* forest);

#endif

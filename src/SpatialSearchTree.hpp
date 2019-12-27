#pragma once

#include "common.hpp"

struct SearchResult {
    IndexT ptIdx; // index of result point in target cloud
    float distSq; // squared distance to the result point
};

class SpatialSearchTree {
public:
    virtual SearchResult NearestSearch(const Point3f& query, float maxDist) const = 0;
    virtual SearchResult ApproxNearestSearch(const Point3f& query) const = 0;
};
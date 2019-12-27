#pragma once

#include "common.hpp"
#include <boost/multi_array.hpp>

#define ROUND(x) (int((x)+0.5))

class LinearDT {
public:
    LinearDT(const PointCloudPtr& cloud, float expandFactor = 2.0f, uint32_t div = 300);
    float Evaluate(Point3f query);

private:
    boost::multi_array<float, 3> grid;
    Bound3f range;
    uint32_t nCells;	// number of cells in one dimension
    float cellEdge;	// namely the length of a cell's edge
};

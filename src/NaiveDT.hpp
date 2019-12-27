#pragma once

#include "KdTree.hpp"
#include <boost/multi_array.hpp>

class NaiveDT {
public:
    NaiveDT(const PointCloudPtr& cloud, uint16_t binSize = 1000, uint32_t div = 300);
    float Evaluate(Point3f query);

private:
    boost::multi_array<float, 3> grid;
    Bound3f range;
    uint32_t nCells[3];
    Vector3f cellSize; // the shape is approximately a cube
};

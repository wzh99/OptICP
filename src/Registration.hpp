#pragma once

#include "common.hpp"

struct RegResult {
    Matrix4f matrix;
    PointCloudT cloud;
    float mse;
};

class Registration {
public:
    virtual void SetTarget(const PointCloudPtr& target) = 0;
    virtual RegResult Register(const PointCloudT& source, const Matrix4f& guess) const = 0;
};
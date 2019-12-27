#pragma once

#include "Registration.hpp"
#include "KdTree.hpp"

class ICP : public Registration {
public:
    ICP(float corresThresh, uint32_t maxIter, float transThresh, float fitThresh,
        uint32_t binCap = 100);
    void SetTarget(const PointCloudPtr& target) override;
    RegResult Register(const PointCloudT& source, const Matrix4f& guess) const override;

private:
    // Reject a correspondence whose distance exceeds this threshhold
    float corresThresh;

    // Maximum iterations to perform
    uint32_t maxIter;

    // Terminate if squared distance of translation components of two consecutive transformations is 
    // below this threshold
    float transThresh;
    // Terminate if MSE between transformed and target point clouds is below this threshold
    float fitThresh;

    // Target cloud pointer
    PointCloudPtr target;

    // Bin capacity of kd-tree index
    uint32_t binCap;
    // K-d tree for target cloud
    std::unique_ptr<KdTree> tree = nullptr;
};
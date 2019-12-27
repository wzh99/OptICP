#pragma once

#include "Registration.hpp"
#include "KdTree.hpp"

class SingleThreadedICP : public Registration {
public:
    SingleThreadedICP(float corresThresh, uint32_t maxIter, float transThresh, float fitThresh,
        uint32_t binCap = 100);
    void SetTarget(const PointCloudPtr& target) override;
    RegResult Register(const PointCloudT& source, const Matrix4f& guess) const override;

private:
    // reject a correspondence whose distance exceeds this threshhold
    float corresThresh;
    // maximum iterations to perform
    uint32_t maxIter;
    // terminate if squared distance of translation components of two consecutive is below 
    // this threshold
    float transThresh;
    // terminate if MSE between transformed and target point clouds is below this threshold
    float fitThresh;
    // Bin capacity of kd-tree index
    uint32_t binCap;
    // Target cloud pointer
    PointCloudPtr target;
    // K-d tree for target cloud
    std::unique_ptr<KdTree> tree = nullptr;
};
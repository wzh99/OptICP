#pragma once

#include "SpatialSearchTree.hpp"

// A naive implementation of K-d tree to demonstrate the significance of memory optimization
class NaiveKdTree : public SpatialSearchTree {
public:
    NaiveKdTree(const PointCloudPtr& cloud, uint16_t binCap);
    SearchResult NearestSearch(const Point3f& query, float maxDist = INFINITY) const override;
    SearchResult ApproxNearestSearch(const Point3f& query) const override;

private:
    struct Node {
        std::unique_ptr<Node> left, right; // left and right subtree
        int dim; // dimension to split
        IndexT split; // index of point used to split
        std::vector<IndexT> bin; // point indices in leaf node

        bool IsLeaf() const { return split == INVALID_INDEX; }
    };

    std::unique_ptr<Node> build(std::vector<IndexT>& indices, IndexT begin, IndexT end,
        const Bound3f& bnd);
    void nearestSearch(const Point3f& query, const Node* node, IndexT& nearest, 
        float& minDistSq) const;

    const PointCloudPtr cloud;
    uint16_t binCap;
    Bound3f cloudBnd;
    std::unique_ptr<Node> root;
    uint32_t height;
};
#pragma once

#include "SpatialSearchTree.hpp"
#include <boost/align/aligned_allocator.hpp>
#include <boost/pool/pool.hpp>

class KdTree : public SpatialSearchTree {
public:
    KdTree(const PointCloudPtr& cloud, uint32_t binCap);
    SearchResult NearestSearch(const Point3f& query, float maxDist = INFINITY) const override;
    SearchResult ApproxNearestSearch(const Point3f& query) const override;

private:
    struct Node { // store one point with each node
        static constexpr int32_t LEAF = -2;
        union {
            struct { // interior node
                int32_t dim; // which dimension to split
                IndexT splitPt; // index of point used to split
                float splitVal; // split value of current dimension        
                IndexT right; // index of right node in node vector
            };
            struct { // leaf node
                IndexT* bin; // pointer to bin, can only be freed manually
                uint32_t size; // number of points stored in current bin
                int32_t type; // leaf node indicator
            };
        }; // 16 bytes in total

        static Node CreateInterior(int16_t dim, IndexT splitPt, float splitVal, IndexT right) {
            Node node;
            node.dim = dim;
            node.splitPt = splitPt;
            node.splitVal = splitVal;
            node.right = right;
            return node;
        }

        static Node CreateLeaf(IndexT* bin, uint32_t size) {
            Node node;
            node.type = LEAF;
            node.bin = bin;
            node.size = size;
            return node;
        }
    };

    using AlignedNodeVector = std::vector<Node, boost::alignment::aligned_allocator<Node, 64>>;

    IndexT build(std::vector<IndexT>& indices, IndexT begin, IndexT end, const Bound3f& ptsBnd);
    void nearestSearch(const Point3f& query, IndexT nodeIdx, IndexT& nearest, float& minDistSq) const;

    const PointCloudPtr cloud;
    uint32_t binCap;
    Bound3f cloudBnd; // axis-aligned bounding box represented by two points
    AlignedNodeVector nodes; // aligned memory to store tree nodes
    boost::pool<> pool; // memory pool to hold bins in leaves
    uint32_t height; // height of K-d tree
};

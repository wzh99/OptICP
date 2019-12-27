#include "KdTree.hpp"
#include <stack>

KdTree::KdTree(const PointCloudPtr& cloud, uint32_t binCap)
    : cloud(cloud), cloudBnd(*cloud), binCap(binCap), pool(sizeof(IndexT)),
    height(uint32_t(std::log2(cloud->size() / binCap)) + 2) {
    // We require bin size to be larger than one to avoid the case where an interior node only 
    // has one child, which adds to complexity of code.
    if (binCap <= 1)
        Error("Bin size for KdTree should be more than one");

    // Initialize point index vector
    std::vector<IndexT> indices(cloud->size());
    for (auto i = 0u; i < cloud->size(); i++)
        indices[i] = i;

    // Build tree
    nodes.reserve(1ull << height);
    build(indices, 0, cloud->size(), cloudBnd);
}

IndexT KdTree::build(std::vector<IndexT>& indices, IndexT begin, IndexT end, const Bound3f& bnd) {
    // Directly store points if under bin capacity
    auto size = end - begin;
    auto curIdx = nodes.size();
    if (size <= binCap) {
        auto bin = static_cast<IndexT*>(pool.ordered_malloc(size));
        for (auto i = begin; i < end; i++)
            bin[i - begin] = indices[i];
        nodes.push_back(Node::CreateLeaf(bin, size));
        return curIdx;
    }

    // Split points by axis with maximum extent
    auto dim = MaxDim(bnd.Diagonal());
    auto mid = (begin + end) / 2;
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
        [&] (IndexT p1, IndexT p2) {
            return cloud->at(p1).data[dim] < cloud->at(p2).data[dim];
        });
    auto splitPt = indices[mid];
    auto splitVal = cloud->at(splitPt).data[dim];
    nodes.push_back(Node::CreateInterior(dim, splitPt, splitVal, INVALID_INDEX));

    // Build left node
    auto leftBnd = bnd;
    leftBnd.max.data[dim] = splitVal;
    build(indices, begin, mid, leftBnd);

    // Build right node
    auto rightBnd = bnd;
    rightBnd.min.data[dim] = splitVal;
    auto rightIdx = build(indices, mid + 1, end, rightBnd);
    nodes[curIdx].right = rightIdx;

    return curIdx;
}

SearchResult KdTree::NearestSearch(const Point3f& query, float maxDist) const {
    auto nearest = INVALID_INDEX;
    auto minDistSq = SQ(maxDist);
    nearestSearch(query, 0, nearest, minDistSq);
    if (nearest == INVALID_INDEX)
        return { INVALID_INDEX, INFINITY };
    return { nearest, minDistSq };
}

void KdTree::nearestSearch(const Point3f& query, IndexT nodeIdx, IndexT& nearest,
    float& minDistSq) const {
    // Define stack
    auto path = static_cast<IndexT*>(alloca(height * sizeof(IndexT)));
    auto len = 0; // length of current path
#define PUSH(x) path[len++] = (x)
#define POP() len--
#define BACK() path[len - 1]
#define EMPTY() (len == 0)

    // Find the region where this query lies using depth first search
    PUSH(nodeIdx); // push index of node passed in
    while (true) {
        auto curIdx = BACK(); // check last node on path
        auto& curNode = nodes[curIdx];
        if (curNode.type == Node::LEAF) break; // leaf node reached, cannot search deeper
        if (query.data[curNode.dim] <= curNode.splitVal)
            PUSH(curIdx + 1); // push left node to path
        else
            PUSH(curNode.right); // push right node to path
    }

    // Test against all points in current leaf node
    auto update = [&] (IndexT idx) {
        auto curDistSq = DistSq(query, cloud->at(idx));
        if (curDistSq < minDistSq) {
            nearest = idx;
            minDistSq = curDistSq;
        }
    };
    auto& leafNode = nodes[BACK()]; // pick leaf node from path
    POP(); // pop leaf node
    for (auto i = 0u; i < leafNode.size; i++)
        update(leafNode.bin[i]);

    // Backtrack to find nearest point
    while (true) {
        // Pop deepest node from path
        if (EMPTY()) break;
        auto curIdx = BACK();
        auto& curNode = nodes[curIdx];
        POP();

        // Test search sphere against splitting plane
        auto center = query.data[curNode.dim];
        auto distSqToPlane = SQ(center - curNode.splitVal);
        if (minDistSq < distSqToPlane) continue; // no need to visit the other node

        // Search in the other side of plane if possible
        update(curNode.splitPt); // point used to split space could be a candidate
        if (center >= curNode.splitVal) // on right side of current node
            nearestSearch(query, curIdx + 1, nearest, minDistSq);
        if (center <= curNode.splitVal) // left side
            nearestSearch(query, curNode.right, nearest, minDistSq);
    }
#undef PUSH
#undef POP
#undef BACK
#undef EMPTY
}

SearchResult KdTree::ApproxNearestSearch(const Point3f& query) const {
    // Find leaf node that contains query point
    auto curIdx = 0u;
    while (true) {
        auto& node = nodes[curIdx];
        if (node.type == Node::LEAF) break; // leaf node found
        if (query.data[node.dim] <= node.splitVal)
            curIdx++;
        else
            curIdx = node.right;
    }
    auto& curNode = nodes[curIdx];

    // Find nearest point in current bin
    auto nearest = INVALID_INDEX;
    auto minDistSq = INFINITY;
    for (auto i = 0u; i < curNode.size; i++) {
        auto curPt = curNode.bin[i];
        auto curDistSq = DistSq(query, cloud->at(curPt));
        if (curDistSq < minDistSq) {
            nearest = curPt;
            minDistSq = curDistSq;
        }
    }

    return { nearest, minDistSq };
}

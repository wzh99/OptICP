#include "NaiveKdTree.hpp"
#include <stack>

NaiveKdTree::NaiveKdTree(const PointCloudPtr& cloud, uint16_t binSize)
    : cloud(cloud), cloudBnd(*cloud), binCap(binSize),
    height(uint32_t(std::log2(cloud->size() / binCap)) + 2) {
    if (binSize <= 1)
        Error("Bin size for KdTree should be more than one");

    // Initialize point index vector, which will be reordered during tree construction
    std::vector<IndexT> indices(cloud->size());
    for (auto i = 0u; i < cloud->size(); i++)
        indices[i] = i;

    // Build tree
    root = build(indices, 0, cloud->size(), cloudBnd);
}

std::unique_ptr<NaiveKdTree::Node>
NaiveKdTree::build(std::vector<IndexT>& indices, IndexT begin, IndexT end, const Bound3f& bnd) {
    // Directly store points if under bin capacity
    auto size = end - begin;
    if (size < binCap) {
        std::vector<IndexT> bin;
        bin.reserve(size);
        for (auto i = begin; i < end; i++)
            bin.push_back(indices[i]);
        return std::make_unique<Node>(Node{ nullptr, nullptr, 0, INVALID_INDEX, std::move(bin) });
    }

    // Split points by axis with maximum extent
    auto dim = MaxDim(bnd.Diagonal());
    auto mid = (begin + end) / 2;
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
        [&] (IndexT i1, IndexT i2) {
        return cloud->at(i1).data[dim] < cloud->at(i2).data[dim];
    });

    // Compute bounds for children
    auto split = cloud->at(indices[mid]).data[dim];
    auto leftBnd = bnd, rightBnd = bnd;
    leftBnd.max.data[dim] = rightBnd.min.data[dim] = split;

    return std::make_unique<Node>(Node{
        build(indices, begin, mid, leftBnd), build(indices, mid + 1, end, rightBnd),
        dim, indices[mid], {},
        });
}

SearchResult NaiveKdTree::NearestSearch(const Point3f& query, float maxDist) const {
    auto nearest = INVALID_INDEX;
    auto minDistSq = SQ(maxDist);
    nearestSearch(query, root.get(), nearest, minDistSq);
    if (nearest == INVALID_INDEX)
        return { INVALID_INDEX, INFINITY };
    return { nearest, minDistSq };
}

void NaiveKdTree::nearestSearch(const Point3f& query, const Node* node, IndexT& nearest,
    float& minDistSq) const {
    // Find the region where this query lies using depth first search
    std::vector<const Node*> path;
    path.reserve(height);
    path.push_back(node);
    while (true) {
        auto curNode = path.back();
        if (curNode->IsLeaf()) break;
        if (query.data[curNode->dim] <= cloud->at(curNode->split).data[curNode->dim])
            path.push_back(curNode->left.get());
        else
            path.push_back(curNode->right.get());
    }

    // Test against all points in current leaf node
    auto update = [&] (IndexT idx) {
        auto curDistSq = DistSq(query, cloud->at(idx));
        if (curDistSq < minDistSq) {
            nearest = idx;
            minDistSq = curDistSq;
        }
    };
    auto leafNode = path.back();
    path.pop_back();
    for (auto idx : leafNode->bin)
        update(idx);

    // Backtrack to find nearest point
    while (true) {
        // Pop deepest node from path
        if (path.empty()) break;
        auto curNode = path.back();
        path.pop_back();

        // Test search sphere against clipping plane
        auto center = query.data[curNode->dim];
        auto splitVal = cloud->at(curNode->split).data[curNode->dim];
        auto distSqToPlane = SQ(center - splitVal);
        if (minDistSq < distSqToPlane) continue; // no need to visit the other node

        // Search in the other side of plane if possible
        update(curNode->split);
        if (center >= splitVal) // on right side of current node
            nearestSearch(query, curNode->left.get(), nearest, minDistSq);
        if (center <= splitVal) // on left side of current node
            nearestSearch(query, curNode->right.get(), nearest, minDistSq);
    }
}

SearchResult NaiveKdTree::ApproxNearestSearch(const Point3f& query) const {
    // Find leaf node that contains query point
    auto curNode = root.get();
    while (true) {
        if (curNode->IsLeaf()) break;
        auto splitVal = cloud->at(curNode->split).data[curNode->dim];
        if (query.data[curNode->dim] <= splitVal)
            curNode = curNode->left.get();
        else
            curNode = curNode->right.get();
    }

    // Find nearest point in current bin
    auto nearest = INVALID_INDEX;
    auto minDistSq = INFINITY;
    for (auto idx : curNode->bin) {
        auto curDistSq = DistSq(query, cloud->at(idx));
        if (curDistSq < minDistSq) {
            nearest = idx;
            minDistSq = curDistSq;
        }
    }

    return { nearest, minDistSq };
}
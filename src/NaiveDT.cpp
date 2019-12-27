#include "NaiveDT.hpp"

NaiveDT::NaiveDT(const PointCloudPtr& cloud, uint16_t binSize, uint32_t div)
    : range(*cloud) {
    // Build distance field's own version of Kd-tree
    KdTree tree(cloud, binSize);

    // Compute range of distance field from bounding box
    auto delta = .1f * MaxComp(range.Diagonal());
    auto expand = Vector3f(delta, delta, delta); // expand range a little
    range.min = range.min - expand;
    range.max = range.max + expand;

    // Compute cell number and size in each dimension
    auto diag = range.Diagonal();
    auto width = MaxComp(diag) / (div - 1);
    for (auto i = 0u; i < 3; i++)
        nCells[i] = diag[i] / width;
    cellSize = Vector3f(diag[0] / nCells[0], diag[1] / nCells[1], diag[2] / nCells[2]);

    // Construct grid
    grid.resize(boost::extents[size_t(nCells[0]) + 1][size_t(nCells[1]) + 1][size_t(nCells[2]) + 1]);
    concurrency::parallel_for(0u, nCells[0] + 1, [&] (IndexT x) {
        for (auto y = 0u; y < nCells[1] + 1; y++)
            for (auto z = 0u; z < nCells[2] + 1; z++) {
                auto pos = range.min + Vector3f(cellSize[0] * x, cellSize[1] * y, cellSize[2] * z);
                grid[x][y][z] = std::sqrt(tree.NearestSearch(pos).distSq);
            }
    });
}

float NaiveDT::Evaluate(Point3f query) {
    // Clamp if query drops out of bound and compute its distance to bound
    auto clamped = Vector3f(Vector3f::Zero());
    for (auto i = 0u; i < 3; i++) {
        if (query.data[i] < range.min.data[i]) {
            clamped[i] = range.min.data[i] - query.data[i];
            query.data[i] = range.min.data[i];
        } else if (query.data[i] > range.max.data[i]) {
            clamped[i] = query.data[i] - range.max.data[i];
            query.data[i] = range.max.data[i];
        }
    }

    // Compute discrete coordinate and interpolation ratio in the grid
    auto relPos = query - range.min; // relative position of grid origin
    Vector3i coord;
    Vector3f ratio;
    for (auto i = 0u; i < 3; i++) {
        coord[i] = std::min(IndexT(relPos[i] / cellSize[i]), nCells[i] - 1);
        ratio[i] = (relPos[i] - coord[i] * cellSize[i]) / cellSize[i];
    }

    // Perform trilinear interpolation on the point
    auto lerp = [] (float t, float v1, float v2) { return (1 - t) * v1 + t * v2; };
    auto lookup = [&] (const Eigen::Vector3i& idx) { return grid[idx[0]][idx[1]][idx[2]]; };
    auto d00 = lerp(ratio[0], lookup(coord), lookup(coord + Vector3i(1, 0, 0)));
    auto d10 = lerp(ratio[0], lookup(coord + Vector3i(0, 1, 0)), lookup(coord + Vector3i(1, 1, 0)));
    auto d01 = lerp(ratio[0], lookup(coord + Vector3i(0, 0, 1)), lookup(coord + Vector3i(1, 0, 1)));
    auto d11 = lerp(ratio[0], lookup(coord + Vector3i(0, 1, 1)), lookup(coord + Vector3i(1, 1, 1)));
    auto d0 = lerp(ratio[1], d00, d10);
    auto d1 = lerp(ratio[1], d01, d11);

    return lerp(ratio[2], d0, d1) + clamped.norm();
}
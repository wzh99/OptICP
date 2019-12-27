#include "LinearDT.hpp"

LinearDT::LinearDT(const PointCloudPtr& cloud, float expandFactor, uint32_t div)
    : range(*cloud) {
    // Compute range of distance field from bounding box
    auto delta = expandFactor * 0.5f * MaxComp(range.Diagonal());
    auto halfDiag = Vector3f(delta, delta, delta);
    auto center = Point3f((range.max.x + range.min.x) / 2, (range.max.y + range.min.y) / 2,
        (range.max.z + range.min.z) / 2);
    range.min = center - halfDiag;
    range.max = center + halfDiag;

    // Compute cell number and size in each dimension
    nCells = div - 1;	// if we have 300 points, then there are 299 cells inside
    cellEdge = 2 * delta / float(nCells);

    // Construct grid and temporary variables
    boost::multi_array<float, 3> g_scan;
    grid.resize(boost::extents[div][div][div]);
    g_scan.resize(boost::extents[div][div][div]);

    // g_scan takes the pointcloud and round them to integer grid coordinates
    // g_scan[i][j][k]=1 if there is a point rounded to the grid position i,j,k.
    // it now acts like the input and grid as the output.
    // since the g function is used to pass former result 2 times, the first pass take g_scan as input
    // and grid as output; second pass reverses that process, and the third pass again reverses it, so 
    // the final result, namely DT, is stored in grid.
    concurrency::parallel_for(0, int(cloud->points.size()), [&] (int index) {
        auto relPos = cloud->points[index] - range.min; // relative position of grid origin
        g_scan[ROUND(relPos[0] / cellEdge)][ROUND(relPos[1] / cellEdge)][ROUND(relPos[2] / cellEdge)]
            = 1.0;
    });

    float infinity = 3 * div;

    // First scan (along dimension z)
    // Grid acts like the g function during the first-dimension scan in the paper
    concurrency::parallel_for(0u, div, [&] (int x) {
        for (auto y = 0u; y < div; y++) {
            /* scan1 */
            if (g_scan[x][y][0] > 0.5)	//not zero
                grid[x][y][0] = 0.0;
            else
                grid[x][y][0] = infinity;

            for (auto z = 1u; z < div; z++)
                if (g_scan[x][y][z] > 0.5)
                    grid[x][y][z] = 0.0;
                else
                    grid[x][y][z] = 1 + grid[x][y][z - 1];

            /* scan2 */
            for (int _z = nCells - 1; _z >= 0; _z--)
                if (grid[x][y][_z + 1] < grid[x][y][_z])
                    grid[x][y][_z] = 1 + grid[x][y][_z + 1];
        }
    });


    // Second scan
    // In this scan, grid is the input, and g_scan is the output.
    auto f_y = [&] (int xCoord, int zCoord, int yCoord, int ypos) {
        return SQ(yCoord - ypos) + SQ(grid[xCoord][ypos][zCoord]);
    };

    auto Sep_y = [&] (int xCoord, int zCoord, int i, int u) {
        assert(u > i);
        return int((SQ(u) - SQ(i) + SQ(grid[xCoord][u][zCoord]) - SQ(grid[xCoord][i][zCoord]))
            / (2 * (u - i)));
    };

    concurrency::parallel_for(0u, div, [&] (int x) {
        std::vector<int> s(div), t(div);

        for (auto z = 0; z < div; z++) {
            auto q = 0;
            s[0] = t[0] = 0;

            for (auto u = 1; u < div; u++) {
                while ((q >= 0) && (f_y(x, z, t[q], s[q]) > f_y(x, z, t[q], u)))
                    q--;

                if (q < 0) {
                    q = 0;
                    s[0] = u;
                } else {
                    auto w = 1 + Sep_y(x, z, s[q], u);
                    if (w < div) {
                        q++;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }

            for (int u = div - 1; u >= 0; u--) {
                auto point = f_y(x, z, u, s[q]);
                g_scan[x][u][z] = point;
                if (u == t[q])
                    q--;
            }
        }
    });

    // Third scan
    // This scan is almost identical to scan2.
    // Takes g_scan as input, and grid as the output
    auto f_x = [&] (int yCoord, int zCoord, int xCoord, int xpos) {
        return SQ(xCoord - xpos) + g_scan[xpos][yCoord][zCoord];
    };

    auto Sep_x = [&] (int yCoord, int zCoord, int i, int u) {
        assert(u > i);
        return int((SQ(u) - SQ(i) + g_scan[u][yCoord][zCoord] - g_scan[i][yCoord][zCoord])
            / (2 * (u - i)));
    };

    concurrency::parallel_for(0u, div, [&] (int y) {
        std::vector<int> s(div), t(div);
        for (auto z = 0; z < div; z++) {
            auto q = 0;
            s[0] = t[0] = 0;

            for (auto u = 1; u < div; u++) {
                while ((q >= 0) && (f_x(y, z, t[q], s[q]) > f_x(y, z, t[q], u)))
                    q--;

                if (q < 0) {
                    q = 0;
                    s[0] = u;
                } else {
                    auto w = 1 + Sep_x(y, z, s[q], u);
                    if (w < div) {
                        q++;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }

            for (auto u = div - 1; u >= 0; u--) {
                auto point = sqrt(f_x(y, z, u, s[q]));
                grid[u][y][z] = point;
                if (u == t[q])
                    q--;
            }
        }
    });
}

float LinearDT::Evaluate(Point3f query) {
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
    auto lookup = [&] (const Eigen::Vector3i& idx) { return grid[idx[0]][idx[1]][idx[2]] * cellEdge; };
    for (auto i = 0u; i < 3; i++)
        coord[i] = std::min(IndexT(relPos[i] / cellEdge + 0.5f), nCells);

    return lookup(coord) + clamped.norm();
}
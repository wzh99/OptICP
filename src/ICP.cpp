#include "ICP.hpp"
#include <functional>

//#define ICP_VERBOSE

ICP::ICP(float corresThresh, uint32_t maxIter, float transThresh, float fitThresh, uint32_t binCap)
    : corresThresh(corresThresh), maxIter(maxIter), transThresh(transThresh), fitThresh(fitThresh),
    binCap(binCap) {}

void ICP::SetTarget(const PointCloudPtr& target) {
    this->target = target;
    tree = std::make_unique<KdTree>(target, binCap);
}

RegResult ICP::Register(const PointCloudT& source, const Matrix4f& guess) const {
    // Initialize result variables
    PointCloudT result(source);
    auto finalMat = guess;
    auto mse = INFINITY;

    for (auto iter = 0u; iter < maxIter; iter++) {
        // Define statistics struct for computing centroid and demean
        struct CorresStat {
            IndexT valid; // whether a point is valid
            float sqErr; // squared distance to nearest point
            Vector3f srcPt, tgtPt; // point data
        };
        const auto emptyStat = CorresStat{ 0, INFINITY, Vector3f::Zero(), Vector3f::Zero() };

        auto nPts = IndexT(source.size());
        std::vector<CorresStat> stat(nPts);
        concurrency::parallel_for(0u, nPts, [&] (IndexT srcIdx) {
            // Find correspondences between current intermediate and target cloud
            auto searchRes = tree->NearestSearch(result[srcIdx], corresThresh);
            if (searchRes.ptIdx == INVALID_INDEX) {
                stat[srcIdx] = emptyStat;
                return;
            }
            // Compute statistical information of current point
            auto srcVec = ToVector3f(result[srcIdx]);
            auto tgtVec = ToVector3f(target->at(searchRes.ptIdx));
            stat[srcIdx] = { 1, searchRes.distSq, srcVec, tgtVec };
        });

        // Accumulate point statistics to compute error and centroid
        auto accum = concurrency::parallel_reduce(stat.begin(), stat.end(), emptyStat,
            [&] (const CorresStat& l, const CorresStat& r) -> CorresStat {
            auto lErr = l.valid ? l.sqErr : 0;
            auto rErr = r.valid ? r.sqErr : 0;
            return { l.valid + r.valid, lErr + rErr, l.srcPt + r.srcPt, l.tgtPt + r.tgtPt };
        });
        auto nValid = accum.valid; // the same in target accumulation result
        mse = accum.sqErr / nValid;
        if (mse < fitThresh) break;
        auto srcCent = Vector3f(accum.srcPt / nValid);
        auto tgtCent = Vector3f(accum.tgtPt / nValid);

        // Compute 3 * 3 matrix
        std::vector<Matrix3f> mat(nPts);
        concurrency::parallel_transform(stat.begin(), stat.end(), mat.begin(),
            [&] (const CorresStat& stat) -> Matrix3f {
            if (!stat.valid) return Matrix3f::Zero();
            auto srcDemean = stat.srcPt - srcCent;
            auto tgtDemean = stat.tgtPt - tgtCent;
            return srcDemean * tgtDemean.transpose();
        });

        // Use SVD to compute transformation matrix
        auto H = concurrency::parallel_reduce(mat.begin(), mat.end(), Matrix3f::Zero(),
            std::plus<Matrix3f>()); // correlation matrix
        Eigen::JacobiSVD<Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto U = svd.matrixU();
        auto V = svd.matrixV();
        if (U.determinant() * V.determinant() < 0)
            for (auto i = 0u; i < 3; i++) V(i, 2) *= -1; // prevent reflection
        auto R = V * U.transpose(); // rotation matrix
        auto curMat = Matrix4f(Matrix4f::Identity());
        curMat.topLeftCorner(3, 3) = R;
        auto Rc = R * srcCent;
        curMat.block(0, 3, 3, 1) = tgtCent - Rc;

        // Update final transformation and apply transformation to intermediate cloud
        auto prevMat = finalMat;
        finalMat = curMat * finalMat;
        concurrency::parallel_for(0u, nPts, [&] (IndexT srcIdx) {
            auto& srcPt = source[srcIdx];
            auto vec = Eigen::Vector4f(finalMat * Eigen::Vector4f(srcPt.x, srcPt.y, srcPt.z, 1));
            result[srcIdx] = Point3f(vec.x(), vec.y(), vec.z());
        });

#ifdef ICP_VERBOSE
        // Print result of this iteration
        std::cout << "Iteration " << iter << '\n';
        std::cout << "Correspondences: " << IndexT(nValid) << '\n';
        std::cout << "MSE: " << mse << '\n';
        std::cout << "Matrix: " << '\n' << finalMat << "\n\n";
#endif // ICP_VERBOSE

        // Early exit if squared difference in translation is below threshold
        // We only need to care about translation difference because translation is 
        // computed after rotation is determined. If translation component changes 
        // little, we can be sure that the algorithm converges.
        if ((finalMat.block(0, 3, 3, 1) - prevMat.block(0, 3, 3, 1)).squaredNorm() < transThresh)
            break;
    }

    return { finalMat, std::move(result), mse };
}

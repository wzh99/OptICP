#include "SingleThreadedICP.hpp"

//#define ICP_VERBOSE

SingleThreadedICP::SingleThreadedICP(float corresThresh, uint32_t maxIter, float transThresh,
    float fitThresh, uint32_t binCap)
    : corresThresh(corresThresh), maxIter(maxIter), transThresh(transThresh), fitThresh(fitThresh),
    binCap(binCap) {}

void SingleThreadedICP::SetTarget(const PointCloudPtr& target) {
    this->target = target;
    tree = std::make_unique<KdTree>(target, binCap);
}

RegResult SingleThreadedICP::Register(const PointCloudT& source, const Matrix4f& guess) const {
    // Initialize result variables
    PointCloudT result(source);
    auto finalMat = guess;
    auto mse = INFINITY;

    for (auto iter = 0u; iter < maxIter; iter++) {
        // Find correspondences and compute statistical data
        struct CorresStat {
            Vector3f srcPt, tgtPt;
        };

        auto nPts = IndexT(source.size());
        std::vector<CorresStat> stat;
        stat.reserve(nPts);
        auto nCorres = 0u;
        auto err = 0.f;
        auto srcCent = Vector3f(Vector3f::Zero()), tgtCent = Vector3f(Vector3f::Zero());

        for (const auto& pt : result) {
            auto searchRes = tree->NearestSearch(pt, corresThresh);
            if (searchRes.ptIdx == INVALID_INDEX) continue;
            nCorres++;
            err += searchRes.distSq;

            auto srcVec = ToVector3f(pt);
            auto tgtVec = ToVector3f(target->at(searchRes.ptIdx));
            srcCent += srcVec;
            tgtCent += tgtVec;
            stat.push_back({ srcVec, tgtVec });
        }

        mse = err / nCorres;
        if (mse < fitThresh) break;
        srcCent /= nCorres;
        tgtCent /= nCorres;

        // Compute 3 * 3 matrix
        auto H = Matrix3f(Matrix3f::Zero());
        for (const auto& c : stat) {
            auto srcDemean = c.srcPt - srcCent;
            auto tgtDemean = c.tgtPt - tgtCent;
            H += srcDemean * tgtDemean.transpose();
        }

        // Use SVD to compute transformation matrix
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
        for (auto i = 0u; i < nPts; i++) {
            auto& srcPt = source[i];
            auto vec = Eigen::Vector4f(finalMat * Eigen::Vector4f(srcPt.x, srcPt.y, srcPt.z, 1));
            result[i] = Point3f(vec.x(), vec.y(), vec.z());
        }

#ifdef ICP_VERBOSE
        // Print result of this iteration
        std::cout << "Iteration " << iter << '\n';
        std::cout << "Correspondences: " << IndexT(nCorres) << '\n';
        std::cout << "MSE: " << mse << '\n';
        std::cout << "Matrix: " << '\n' << finalMat << "\n\n";
#endif // ICP_VERBOSE

        // Early exit if squared difference in translation is below threshold
        // We only need to care about translation difference because translation is 
        // computed after rotation is determined. If translation component changes 
        // little, we can be sure that the algorithm converges.
        if ((finalMat.block(0, 3, 3, 1) - prevMat.block(0, 3, 3, 1)).squaredNorm()
            < transThresh)
            break;
    }

    return { finalMat, std::move(result), mse };
}
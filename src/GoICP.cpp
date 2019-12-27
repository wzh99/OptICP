#include "GoICP.hpp"

GoICP::GoICP(/* ICP parameters */ float corresThresh, uint32_t maxIter, float transThresh,
    float fitThresh, uint32_t bin_size,
    /* GoICP parameters */ float mseThresh, float rotMinX, float rotMinY, float rotMinZ, float rotWidth,
    float transMinX, float transMinY, float transMinZ, float transWidth, float trimFraction,
    float expandFactor, uint32_t div)
    : icp(corresThresh, maxIter, transThresh, fitThresh, bin_size), binSize(bin_size),
    mseThresh(mseThresh), trimFraction(trimFraction), doTrim(trimFraction > 0.001),
    expandFactor(expandFactor), div(div) {
    initNodeRot.a = rotMinX;
    initNodeRot.b = rotMinY;
    initNodeRot.c = rotMinZ;
    initNodeRot.w = rotWidth;
    initNodeTrans.x = transMinX;
    initNodeTrans.y = transMinY;
    initNodeTrans.z = transMinZ;
    initNodeTrans.w = transWidth;
    initNodeRot.l = 0;
    initNodeRot.lb = 0;
    initNodeTrans.lb = 0;
}

// Cloud setting (make sure to call it before DT building!)
void GoICP::SetSource(PointCloudPtr src) {
    pData = src;
}

void GoICP::SetTarget(PointCloudPtr tgt) {
    pModel = tgt;
    icp.SetTarget(pModel);
    ldt = std::make_unique<LinearDT>(pModel, expandFactor, div);
}

const int GoICP::GetModelSize() {
    return pModel->points.size();
}

const int GoICP::GetDataSize() {
    return pData->points.size();
}

const Matrix4f GoICP::GetOptMat() {
    return optMat;
}

// Run ICP and calculate sum squared L2 error
float GoICP::runICP(Matrix4f& trans_now) {
    auto icp_result = icp.Register(*pData, trans_now);
    trans_now = icp_result.matrix;

    // Transform point cloud and use DT to determine the L2 error
    std::vector<float> error(icp_result.cloud.points.size());
    concurrency::parallel_for(0, int(pData->points.size()), [&] (int i) {
        Point3f& ptemp = icp_result.cloud.points[i];
        float dis = ldt->Evaluate(ptemp);
        error[i] = dis * dis;
    });

    if (doTrim) {
        // Sort error in ascending order
        intro_select(error, 0, pData->points.size() - 1, inlierNum - 1);
    }

    float SSE = 0.0f;
    for (int i = 0; i < inlierNum; i++)
        SSE += error[i];

    return SSE;
}

void GoICP::initialize() {
    int i, j;
    float sigma, maxAngle;
    auto normData = std::vector<float>(pData->points.size());

    // Precompute the rotation uncertainty distance (maxRotDis) for each point in the data 
    // and each level of rotation subcube
    // Calculate L2 norm of each point in data cloud to origin
    for (i = 0; i < pData->points.size(); i++) {
        normData[i] = sqrt(SQ(pData->points[i].x) + SQ(pData->points[i].y) + SQ(pData->points[i].z));
    }

    maxRotDis = new float* [MAXROTLEVEL];
    for (i = 0; i < MAXROTLEVEL; i++) {
        maxRotDis[i] = (float*) malloc(sizeof(float*) * pData->points.size());

        sigma = initNodeRot.w / pow(2.0, i) / 2.0; // Half-side length of each level of rotation subcube
        maxAngle = SQRT3 * sigma;

        if (maxAngle > PI)
            maxAngle = PI;
        for (j = 0; j < pData->points.size(); j++)
            maxRotDis[i][j] = 2 * sin(maxAngle / 2) * normData[j];
    }

    // Temporary Variable
    // we declare it here because we don't want these space to be allocated and deallocated 
    // again and again each time inner BNB runs.
    minDis = std::vector<float>(pData->points.size());

    // Initialise so-far-best rotation and translation nodes
    optNodeRot = initNodeRot;
    optNodeTrans = initNodeTrans;
    // Initialise so-far-best rotation and translation matrix
    optMat = Eigen::Matrix4f::Identity();

    // For untrimmed ICP, use all points, otherwise only use inlierNum points
    if (doTrim) {
        // Calculate number of inlier points
        inlierNum = (int) (pData->points.size() * (1 - trimFraction));
    } else {
        inlierNum = pData->points.size();
    }
    sseThresh = mseThresh * inlierNum;
}

void GoICP::clear() {
    for (int i = 0; i < MAXROTLEVEL; i++)
        delete(maxRotDis[i]);
    delete(maxRotDis);
}

// Inner Branch-and-Bound, iterating over the translation space
float GoICP::innerBnB(float* maxRotDisL, TranslationNode* nodeTransOut,
    const std::vector<Point3f>& pDataTemp) {
    int j;
    float transX, transY, transZ;
    float lb, ub, optErrorT;
    float maxTransDis;
    TranslationNode nodeTrans, nodeTransParent;
    priority_queue<TranslationNode> queueTrans;

    // Set optimal translation error to overall so-far optimal error
    // Investigating translation nodes that are sub-optimal overall is redundant
    optErrorT = optError;

    // Push top-level translation node into the priority queue
    queueTrans.push(initNodeTrans);

    while (1) {
        if (queueTrans.empty())
            break;

        cout << '.';
        fflush(stdout);

        nodeTransParent = queueTrans.top();
        queueTrans.pop();

        if (optErrorT - nodeTransParent.lb < sseThresh) {
            break;
        }

        nodeTrans.w = nodeTransParent.w / 2;
        maxTransDis = SQRT3 / 2.0 * nodeTrans.w;

        for (j = 0; j < 8; j++) {
            nodeTrans.x = nodeTransParent.x + (j & 1) * nodeTrans.w;
            nodeTrans.y = nodeTransParent.y + (j >> 1 & 1)* nodeTrans.w;
            nodeTrans.z = nodeTransParent.z + (j >> 2 & 1)* nodeTrans.w;

            transX = nodeTrans.x + nodeTrans.w / 2;
            transY = nodeTrans.y + nodeTrans.w / 2;
            transZ = nodeTrans.z + nodeTrans.w / 2;

            struct ReduceElement {
                float sqr;
                float sqr_subTransDis;
            };

            auto minDisSqr = std::make_unique<ReduceElement[]>(pData->points.size());

            // For each data point, calculate the distance to it's closest point in the model cloud
            concurrency::parallel_for(0, int(pData->points.size()), [&] (int i) {
                // Find distance between transformed point and closest point in model set 
                // ||R_r0 * x + t0 - y||
                // pDataTemp is the data points rotated by R0
                minDis[i] = ldt->Evaluate(Point3f(pDataTemp[i].data[0] + transX,
                    pDataTemp[i].data[1] + transY,
                    pDataTemp[i].data[2] + transZ));

                // Subtract the rotation uncertainty radius if calculating the rotation lower bound
                // maxRotDisL == NULL when calculating the rotation upper bound
                if (maxRotDisL)
                    minDis[i] = max(minDis[i] - maxRotDisL[i], 0.0f);

                if (!doTrim) {
                    minDisSqr[i].sqr = minDis[i] * minDis[i];
                    minDisSqr[i].sqr_subTransDis = max(minDis[i] - maxTransDis, 0.0f);
                    minDisSqr[i].sqr_subTransDis *= minDisSqr[i].sqr_subTransDis;
                }
            });

            if (doTrim) {
                intro_select(minDis, 0, pData->points.size() - 1, inlierNum - 1);
                concurrency::parallel_for(0, inlierNum, [&] (int i) {
                    minDisSqr[i].sqr = minDis[i] * minDis[i];
                    minDisSqr[i].sqr_subTransDis = max(minDis[i] - maxTransDis, 0.0f);
                    minDisSqr[i].sqr_subTransDis *= minDisSqr[i].sqr_subTransDis;
                });
            }

            // For each data point, find the incremental upper and lower bounds
            auto boundAccumFunc = [&] (const ReduceElement& dis1, const ReduceElement& dis2) {
                return ReduceElement{ dis1.sqr + dis2.sqr, dis1.sqr_subTransDis + dis2.sqr_subTransDis };
            };

            auto reduce_res = concurrency::parallel_reduce(&minDisSqr[0], &minDisSqr[inlierNum],
                ReduceElement{ 0.0, 0.0 }, boundAccumFunc);
            ub = reduce_res.sqr;
            lb = reduce_res.sqr_subTransDis;

            // If upper bound is better than best, update optErrorT and optTransOut (optimal 
            // translation node)
            if (ub < optErrorT) {
                optErrorT = ub;
                if (nodeTransOut)
                    *nodeTransOut = nodeTrans;
            }

            // Remove subcube from queue if lb is bigger than optErrorT
            if (lb >= optErrorT) {
                //discard
                continue;
            }

            nodeTrans.ub = ub;
            nodeTrans.lb = lb;
            queueTrans.push(nodeTrans);
        }
    }

    return optErrorT;
}

float GoICP::outerBnB() {
    int i, j;
    RotationNode nodeRot, nodeRotParent;
    TranslationNode nodeTrans;
    float v1, v2, v3, t, ct, ct2, st, st2;
    float tmp121, tmp122, tmp131, tmp132, tmp231, tmp232;
    float R11, R12, R13, R21, R22, R23, R31, R32, R33;
    float lb, ub, error;
    clock_t clockBeginICP;
    priority_queue<RotationNode> queueRot;
    auto pDataTemp = vector<Point3f>(pData->points.size());

    // Calculate Initial Error
    optError = 0;

    for (i = 0; i < pData->points.size(); i++) {
        minDis[i] = ldt->Evaluate(Point3f(pData->points[i]));
    }
    if (doTrim) {
        intro_select(minDis, 0, pData->points.size() - 1, inlierNum - 1);
    }
    for (i = 0; i < inlierNum; i++) {
        optError += minDis[i] * minDis[i];
    }
    cout << "Error*: " << optError << " (Init)" << endl;

    // Temporary matrix storeing both input and output of icp (through citation, &).
    // Because the approximating nature of DT, ICP result may return  an error slightly bigger than 
    // formal optimal one, we cannot directly let ICP change the optimal matrix.
    Matrix4f mat = optMat;

    // Run ICP from initial state
    clockBeginICP = clock();
    error = runICP(mat);
    if (error < optError) {
        optError = error;
        optMat = mat;
        cout << "Error*: " << error << " (ICP " << (double) (clock() - clockBeginICP) /
            CLOCKS_PER_SEC << "s)" << endl;
        cout << "ICP-ONLY Affine Matrix:" << endl;
        cout << mat.matrix() << endl;
    }

    // Push top-level rotation node into priority queue
    queueRot.push(initNodeRot);

    // Keep exploring rotation space until convergence is achieved
    long long count = 0;
    while (1) {
        if (queueRot.empty()) {
            cout << "Rotation Queue Empty" << endl;
            cout << "Error*: " << optError << ", LB: " << lb << endl;
            break;
        }

        cout << 'o';
        fflush(stdout);

        // Access rotation cube with lowest lower bound...
        nodeRotParent = queueRot.top();
        // ...and remove it from the queue
        queueRot.pop();

        // Exit if the optError is less than or equal to the lower bound plus a small epsilon
        if ((optError - nodeRotParent.lb) <= sseThresh) {
            cout << "Error*: " << optError << ", LB: " << nodeRotParent.lb << ", epsilon: "
                << sseThresh << endl;
            break;
        }

        if (count++ % 10 == 0) {
            printf("LB=%f  L=%d  ", nodeRotParent.lb, nodeRotParent.l);
            cout << "optError: " << optError << ", LB: " << lb << endl;
        }

        // Subdivide rotation cube into octant subcubes and calculate upper and lower bounds for each
        nodeRot.w = nodeRotParent.w / 2;
        nodeRot.l = nodeRotParent.l + 1;
        // For each subcube,
        for (j = 0; j < 8; j++) {
            // Calculate the smallest rotation across each dimension
            nodeRot.a = nodeRotParent.a + (j & 1) * nodeRot.w;
            nodeRot.b = nodeRotParent.b + (j >> 1 & 1)* nodeRot.w;
            nodeRot.c = nodeRotParent.c + (j >> 2 & 1)* nodeRot.w;

            // Find the subcube centre
            v1 = nodeRot.a + nodeRot.w / 2;
            v2 = nodeRot.b + nodeRot.w / 2;
            v3 = nodeRot.c + nodeRot.w / 2;

            // Skip subcube if it is completely outside the rotation PI-ball
            if (sqrt(v1 * v1 + v2 * v2 + v3 * v3) - SQRT3 * nodeRot.w / 2 > PI) {
                continue;
            }

            // Convert angle-axis rotation into a rotation matrix
            t = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            if (t > 0) {
                v1 /= t;
                v2 /= t;
                v3 /= t;

                ct = cos(t);
                ct2 = 1 - ct;
                st = sin(t);
                st2 = 1 - st;

                tmp121 = v1 * v2 * ct2; tmp122 = v3 * st;
                tmp131 = v1 * v3 * ct2; tmp132 = v2 * st;
                tmp231 = v2 * v3 * ct2; tmp232 = v1 * st;

                R11 = ct + v1 * v1 * ct2;		R12 = tmp121 - tmp122;		R13 = tmp131 + tmp132;
                R21 = tmp121 + tmp122;		R22 = ct + v2 * v2 * ct2;		R23 = tmp231 - tmp232;
                R31 = tmp131 - tmp132;		R32 = tmp231 + tmp232;		R33 = ct + v3 * v3 * ct2;

                // Rotate data points by subcube rotation matrix
                concurrency::parallel_for(0, int(pData->points.size()), [&] (int i) {
                    Point3f& p = pData->points[i];
                    pDataTemp[i].x = R11 * p.x + R12 * p.y + R13 * p.z;
                    pDataTemp[i].y = R21 * p.x + R22 * p.y + R23 * p.z;
                    pDataTemp[i].z = R31 * p.x + R32 * p.y + R33 * p.z;
                });
            }
            // If t == 0, the rotation angle is 0 and no rotation is required
            else
                concurrency::parallel_for(0, int(pData->points.size()), [&] (int i) {
                pDataTemp[i] = pData->points[i];
            });

            // Upper Bound
            // Run Inner Branch-and-Bound to find rotation upper bound
            // Calculates the rotation upper bound by finding the translation upper bound for a given 
            // rotation, assuming that the rotation is known (zero rotation uncertainty radius)
            ub = innerBnB(NULL /*Rotation Uncertainty Radius*/, &nodeTrans, pDataTemp);

            // If the upper bound is the best so far, run ICP
            if (ub < optError) {
                // Update optimal error and rotation/translation nodes
                optError = ub;
                optNodeRot = nodeRot;
                optNodeTrans = nodeTrans;

                optMat(0, 0) = R11; optMat(0, 1) = R12; optMat(0, 2) = R13;
                optMat(1, 0) = R21; optMat(1, 1) = R22; optMat(1, 2) = R23;
                optMat(2, 0) = R31; optMat(2, 1) = R32; optMat(2, 2) = R33;
                optMat(0, 3) = optNodeTrans.x + optNodeTrans.w / 2;
                optMat(1, 3) = optNodeTrans.y + optNodeTrans.w / 2;
                optMat(2, 3) = optNodeTrans.z + optNodeTrans.w / 2;

                cout << "Error*: " << optError << endl;

                // Run ICP
                clockBeginICP = clock();
                mat = optMat;
                error = runICP(mat);
                // Our ICP implementation uses kdtree for closest distance computation which is slightly 
                // different from DT approximation, thus it's possible that ICP failed to decrease the 
                // DT error. This is no big deal as the difference should be very small.
                if (error < optError) {
                    optError = error;
                    optMat = mat;
                    cout << "Error*: " << error << "(ICP " << (double) (clock() - clockBeginICP) /
                        CLOCKS_PER_SEC << "s)" << endl;
                }

                // Discard all rotation nodes with high lower bounds in the queue
                priority_queue<RotationNode> queueRotNew;
                while (!queueRot.empty()) {
                    RotationNode node = queueRot.top();
                    queueRot.pop();
                    if (node.lb < optError)
                        queueRotNew.push(node);
                    else
                        break;
                }
                queueRot = queueRotNew;
            }

            // Lower Bound
            // Run Inner Branch-and-Bound to find rotation lower bound
            // Calculates the rotation lower bound by finding the translation upper bound for a given 
            // rotation, assuming that the rotation is uncertain (a positive rotation uncertainty radius)
            // Pass an array of rotation uncertainties for every point in data cloud at this level
            lb = innerBnB(maxRotDis[nodeRot.l], nullptr /*Translation Node*/, pDataTemp);

            // If the best error so far is less than the lower bound, remove the rotation subcube from 
            // the queue
            if (lb >= optError) {
                continue;
            }

            // Update node and put it in queue
            nodeRot.ub = ub;
            nodeRot.lb = lb;
            queueRot.push(nodeRot);
        }
    }

    return optError;
}

float GoICP::Register() {
    initialize();
    outerBnB();
    clear();
    return optError;
}

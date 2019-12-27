#include "load.hpp"
#include "KdTree.hpp"
#include "NaiveKdTree.hpp"
#include "ICP.hpp"
#include "SingleThreadedICP.hpp"
#include "GoICP.hpp"
#include "jly_3ddt.h"
#include "LinearDT.hpp"
#include "NaiveDT.hpp"

#include <random>
#include <chrono>
#include <boost/make_shared.hpp>
#include <pcl/common/transforms.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/ia_fpcs.h>

#define DEFAULT_OUTPUT_FNAME "output.txt"
#define DEFAULT_MODEL_FNAME "model_bunny.pcd"
#define DEFAULT_DATA_FNAME "data_bunny.pcd"

//parameters
// ICP
#define CORRES_THRESH 1
#define MAX_ITER 100
#define TRANS_THRESH 0.0
#define FIT_THRESH 0.0
#define BIN_SIZE 100
// GoICP
// Mean Squared Error(MSE) convergence threshold
#define MSE_THRESH 0.001f

// Smallest rotation value along dimension X of rotation cube(radians)
#define ROT_MIN_X -3.1415926536f
// Smallest rotation value along dimension Y of rotation cube(radians)
#define ROT_MIN_Y -3.1415926536f
// Smallest rotation value along dimension Z of rotation cube(radians)
#define ROT_MIN_Z -3.1415926536f
// Side length of each dimension of rotation cube(radians)
#define ROT_WIDTH 6.2831853072f

// Smallest translation value along dimension X of translation cube
#define TRANS_MIN_X -0.3f
// Smallest translation value along dimension Y of translation cube
#define TRANS_MIN_Y -0.3f
// Smallest translation value along dimension Z of translation cube
#define TRANS_MIN_Z -0.3f
// Side length of each dimension of translation cube
#define TRANS_WIDTH 0.6f

// Set to 0.0 for no trimming
#define TRIM_FRACTION 0.0f

#define EXPAND_FACTOR 2
#define DIVIDE 300

using namespace std;
using namespace std::chrono;

IndexT naiveSearch(PointCloudT& cloud, const Point3f& query) {
    auto closest = 0;
    auto minDistSq = INFINITY;
    for (auto i = 0u; i < cloud.size(); i++) {
        auto curDistSq = DistSq(query, cloud[i]);
        if (curDistSq < minDistSq) {
            closest = i;
            minDistSq = curDistSq;
        }
    }
    return closest;
}

void testKdTreeCorrectness(const PointCloudPtr& cloud) {
    KdTree tree(cloud, 100);
    NaiveKdTree Ntree(cloud, 100);
    auto bnd = Bound3f(*cloud);
    auto correct = 0, Ncorrect = 0;
    constexpr auto nTests = 5000;
    std::default_random_engine rng;
    std::uniform_real_distribution<float> distrib;
    for (auto i = 0u; i < nTests; i++) {
        float v[3];
        for (auto j = 0u; j < 3; j++) {
            auto t = distrib(rng);
            v[j] = (1 - t) * bnd.min.data[j] + t * bnd.max.data[j];
        }
        auto query = Point3f(v[0], v[1], v[2]);
        auto result = tree.NearestSearch(query);
        auto Nresult = Ntree.NearestSearch(query);
        auto truth = naiveSearch(*cloud, query);
        correct += result.ptIdx == truth;
        Ncorrect += Nresult.ptIdx == truth;
    }
    std::cout << "correctness: " << float(correct) / nTests << '\n';
    std::cout << "Ncorrectness: " << float(Ncorrect) / nTests << '\n';
}

using namespace std::chrono;

void testKdTreeEfficiency(const PointCloudPtr& cloud, const int binSize) {
    KdTree tree(cloud, binSize);
    NaiveKdTree naive(cloud, binSize);
    auto bnd = Bound3f(*cloud);
    constexpr auto nTests = 10000;
    long oneTime = 0, twoTime = 0;
    std::default_random_engine rng;
    std::uniform_real_distribution<float> distrib;
    auto samples = std::make_unique<Point3f[]>(nTests);
    for (auto i = 0u; i < nTests; i++) {
        for (auto j = 0u; j < 3; j++) {
            auto t = distrib(rng);
            samples[i].data[j] = (1 - t) * bnd.min.data[j] + t * bnd.max.data[j];
        }
    }
    for (auto i = 0u; i < nTests; i++) {
        auto tick = steady_clock::now();
        tree.NearestSearch(samples[i]);
        oneTime += duration_cast<nanoseconds>(steady_clock::now() - tick).count();
    }
    for (auto i = 0u; i < nTests; i++) {
        auto tick = steady_clock::now();
        naive.NearestSearch(samples[i]);
        twoTime += duration_cast<nanoseconds>(steady_clock::now() - tick).count();
    }
    std::cout << oneTime / nTests << ' ' << twoTime / nTests << '\n';
}

void testICP(const PointCloudPtr& original) {
    auto transformed = boost::make_shared<PointCloudT>();
    auto middle = boost::make_shared<PointCloudT>();
    // auto registered = boost::make_shared<PointCloudT>();

    // Compute target tranform matrix
    Point3f min, max;
    pcl::getMinMax3D(*original, min, max);
    auto affine = Eigen::Affine3f::Identity();
    affine.translation() << -(min.x + max.x) / 2, -(min.y + max.y) / 2, -(min.z + max.z) / 2;
    affine.prerotate(Eigen::AngleAxisf(M_PI / 4, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*original, *transformed, affine);

    // Run FPCS for intial alignment
    pcl::registration::FPCSInitialAlignment<Point3f, Point3f> fpcs;
    fpcs.setInputSource(original);
    fpcs.setInputTarget(transformed);
    fpcs.setMaxComputationTime(10);
    fpcs.setNumberOfSamples(200);
    fpcs.align(*middle);
    auto guess = fpcs.getFinalTransformation();

    // Run ICP for precise alignment
    boost::shared_ptr<PointCloudT> registered;
    {
        ICP icp(1e-2, 50, 1e-8, 1e-6, 100);
        icp.SetTarget(transformed);
        auto tick = steady_clock::now();
        auto result = icp.Register(*original, guess);
        auto time = duration_cast<microseconds>(steady_clock::now() - tick).count();
        std::cout << "Multi-threaded ICP Time: " << time << '\n';
        registered = boost::make_shared<PointCloudT>(result.cloud);
    }
    {
        SingleThreadedICP icp(1e-2, 50, 1e-8, 1e-6, 100);
        icp.SetTarget(transformed);
        auto tick = steady_clock::now();
        auto result = icp.Register(*original, guess);
        auto time = duration_cast<microseconds>(steady_clock::now() - tick).count();
        std::cout << "Single-threaded ICP Time: " << time << '\n';
        //registered = boost::make_shared<PointCloudT>(result.cloud);
    }

    // Visualize registration result
    using namespace pcl::visualization;
    using ColorHandler = PointCloudColorHandlerCustom<Point3f>;
    CloudViewer viewer("Cloud viewer");
    viewer.runOnVisualizationThreadOnce([&] (PCLVisualizer& vis) {
        vis.setBackgroundColor(1, 1, 1);
        // ColorHandler origColor(original, 255, 0, 0);
        // vis.addPointCloud(original, origColor, "original");
        ColorHandler transColor(transformed, 0, 255, 0);
        vis.addPointCloud(transformed, transColor, "transformed");
        ColorHandler midColor(middle, 0, 0, 255);
        vis.addPointCloud(middle, midColor, "middle");
        ColorHandler regColor(registered, 255, 0, 255);
        vis.addPointCloud(registered, regColor, "registered");
    });
    while (!viewer.wasStopped());
}


void displayResult(const PointCloudPtr& original, const PointCloudPtr& target, const Matrix4f regResult) {
    auto registered = boost::make_shared<PointCloudT>();

    // transform to get registered cloud
    pcl::transformPointCloud(*original, *registered, regResult);

    // Visualize registration result
    using namespace pcl::visualization;
    using ColorHandler = PointCloudColorHandlerCustom<Point3f>;
    CloudViewer viewer("Cloud viewer");
    viewer.runOnVisualizationThreadOnce([&] (PCLVisualizer& vis) {
        vis.setBackgroundColor(1, 1, 1);
        ColorHandler origColor(original, 255, 0, 0);
        vis.addPointCloud(original, origColor, "original");
        ColorHandler targetColor(target, 0, 255, 0);
        vis.addPointCloud(target, targetColor, "target");
        ColorHandler regColor(registered, 0, 0, 255);
        vis.addPointCloud(registered, regColor, "registered");
    });
    while (!viewer.wasStopped());
}

void testGoICP() {
    clock_t  clockBegin, clockEnd;
    PointCloudPtr pModel, pData;
    pModel = LoadPCDFile(DEFAULT_MODEL_FNAME);
    pData = LoadPCDFile(DEFAULT_DATA_FNAME);

    GoICP goicp(
        /* ICP params */
        CORRES_THRESH,
        MAX_ITER,
        TRANS_THRESH,
        FIT_THRESH,
        BIN_SIZE,
        /* GoICP params */
        MSE_THRESH,
        ROT_MIN_X,
        ROT_MIN_Y,
        ROT_MIN_Z,
        ROT_WIDTH,
        TRANS_MIN_X,
        TRANS_MIN_Y,
        TRANS_MIN_Z,
        TRANS_WIDTH,
        TRIM_FRACTION,
        EXPAND_FACTOR,
        DIVIDE
    );

    goicp.SetSource(pData);

    // Build Distance Transform
    cout << "Building Distance Transform..." << flush;
    clockBegin = clock();
    goicp.SetTarget(pModel);
    clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "s (CPU)" << endl;

    // Run GO-ICP
    cout << "Model ID: " << DEFAULT_MODEL_FNAME << " (" << goicp.GetModelSize()
        << "), Data ID: " << DEFAULT_DATA_FNAME << " (" << goicp.GetDataSize() << ")" << endl;
    cout << "Registering..." << endl;
    clockBegin = clock();
    goicp.Register();
    clockEnd = clock();
    double time = (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC;
    cout << "Optimal Transform Matrix:" << endl;
    cout << goicp.GetOptMat() << endl;
    cout << "Finished in " << time << endl;

    ofstream ofile;
    string outputfilename = DEFAULT_OUTPUT_FNAME;
    ofile.open(outputfilename.c_str(), ofstream::out);
    ofile << time << endl;
    ofile << goicp.GetOptMat() << endl;
    ofile.close();

    //displayResult(pData, pModel, goicp.getOptMat());
}

void testFpcsGoicpImprovement() {
    clock_t  clockBegin, clockEnd;
    PointCloudPtr pModel, pData_original, pData, middle;
    pModel = LoadPCDFile(DEFAULT_MODEL_FNAME);
    pData_original = LoadPCDFile(DEFAULT_DATA_FNAME);
    pData = boost::make_shared<PointCloudT>();

    // translate the target point cloud
    double translateZ = 0.2;
    assert((translateZ + 0.077 < -(TRANS_MIN_Z)) && (translateZ + 0.076 > TRANS_MIN_Z));
    auto affine = Eigen::Affine3f::Identity();
    affine.translation() << 0, 0, translateZ;
    pcl::transformPointCloud(*pData_original, *pData, affine);

    GoICP goicp(
        /* ICP params */
        CORRES_THRESH,
        MAX_ITER,
        TRANS_THRESH,
        FIT_THRESH,
        BIN_SIZE,
        /* GoICP params */
        MSE_THRESH,
        ROT_MIN_X,
        ROT_MIN_Y,
        ROT_MIN_Z,
        ROT_WIDTH,
        TRANS_MIN_X,
        TRANS_MIN_Y,
        TRANS_MIN_Z,
        TRANS_WIDTH,
        TRIM_FRACTION,
        EXPAND_FACTOR,
        DIVIDE
    );

    // Bare Go-ICP
    cout << "Bare GoICP:" << flush;
    goicp.SetSource((PointCloudPtr) pData);
    clockBegin = clock();
    goicp.SetTarget((PointCloudPtr) pModel);
    goicp.Register();
    clockEnd = clock();
    double timeGoICP = (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC;
    cout << "Optimal Transform Matrix:" << endl;
    cout << goicp.GetOptMat() << endl;
    cout << "Finished in totally " << timeGoICP << " seconds" << endl;

    //displayResult(pData, pModel, goicp.getOptMat());

    GoICP goicp2(
        /* ICP params */
        CORRES_THRESH,
        MAX_ITER,
        TRANS_THRESH,
        FIT_THRESH,
        BIN_SIZE,
        /* GoICP params */
        MSE_THRESH,
        ROT_MIN_X,
        ROT_MIN_Y,
        ROT_MIN_Z,
        ROT_WIDTH,
        TRANS_MIN_X,
        TRANS_MIN_Y,
        TRANS_MIN_Z,
        TRANS_WIDTH,
        TRIM_FRACTION,
        EXPAND_FACTOR,
        DIVIDE
    );
    cout << "GoICP with fpcs initial alignment:" << flush;
    clockBegin = clock();
    middle = boost::make_shared<PointCloudT>();
    pcl::registration::FPCSInitialAlignment<Point3f, Point3f> fpcs;
    fpcs.setInputSource(pData);
    fpcs.setInputTarget(pModel);
    fpcs.setApproxOverlap(0.7);
    fpcs.setDelta(0.01);
    fpcs.setMaxComputationTime(1000);
    fpcs.setNumberOfSamples(200);
    fpcs.align(*middle);
    clockEnd = clock();
    timeGoICP = (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC;

    //displayResult(pData, pModel, fpcs.getFinalTransformation());

    goicp2.SetSource((PointCloudPtr) middle);
    clockBegin = clock();
    goicp2.SetTarget((PointCloudPtr) pModel);
    goicp2.Register();
    clockEnd = clock();
    timeGoICP += (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC;
    cout << "Optimal Transform Matrix:" << endl;
    cout << goicp2.GetOptMat() << endl;
    cout << "Finished in totally " << timeGoICP << " seconds" << endl;

    displayResult(pData, pModel, goicp2.GetOptMat() * fpcs.getFinalTransformation());
}

void testdt() {
    clock_t  clockBegin, clockEnd;
    PointCloudPtr pModel;
    DT3D dt;
    pModel = LoadPCDFile(DEFAULT_MODEL_FNAME);
    dt.SIZE = 300;
    dt.expandFactor = 2.0;

    cout << "buiding accurate KdTree:\n";
    KdTree tree(pModel, 100);

    cout << "building dt(s):";
    clockBegin = clock();
    auto x = std::make_unique<double[]>(pModel->size());
    auto y = std::make_unique<double[]>(pModel->size());
    auto z = std::make_unique<double[]>(pModel->size());
    for (int i = 0; i < pModel->points.size(); i++) {
        x[i] = pModel->points[i].x;
        y[i] = pModel->points[i].y;
        z[i] = pModel->points[i].z;
    }
    dt.Build(x.get(), y.get(), z.get(), pModel->points.size());
    clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;

    cout << "building dt(by wzh)(s):";
    clockBegin = clock();
    NaiveDT df(pModel);
    clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;

    cout << "building dt(by lq)(s):";
    clockBegin = clock();
    LinearDT ldt(pModel);
    clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;

    auto bnd = Bound3f(*pModel);
    auto correct = 0;
    constexpr auto nTests = 100;
    auto dtTime = 0, dfTime = 0, ldtTime = 0;
    float diff_y = 0.0, diff_w = 0.0, diff_l = 0.0;
    float diff_yw = 0.0, diff_yl = 0.0;
    std::default_random_engine rng;
    std::uniform_real_distribution<float> distrib;

    for (auto i = 0u; i < nTests; i++) {
        float v[3];
        for (auto j = 0u; j < 3; j++) {
            auto t = distrib(rng);
            v[j] = (1 - t) * bnd.min.data[j] + t * bnd.max.data[j];
        }

        auto query = Point3f(v[0], v[1], v[2]);
        cout << "query: " << query;

        auto tick = steady_clock::now();
        auto err_dt = dt.Distance(v[0], v[1], v[2]);
        dtTime += duration_cast<microseconds>(steady_clock::now() - tick).count();

        cout << " dt: " << err_dt;

        tick = steady_clock::now();
        auto err_df = df.Evaluate(query);
        dfTime += duration_cast<microseconds>(steady_clock::now() - tick).count();

        cout << " dt(by wzh): " << err_df;

        tick = steady_clock::now();
        auto err_ldt = ldt.Evaluate(query);
        ldtTime += duration_cast<microseconds>(steady_clock::now() - tick).count();

        cout << " dt(by lq): " << err_ldt << '\n';

        auto res_tree = tree.NearestSearch(query);
        auto err_accurate = res_tree.distSq;

        diff_y += err_accurate - err_dt;
        diff_w += err_accurate - err_df;
        diff_l += err_accurate - err_ldt;

        diff_yw += err_dt - err_df;
        diff_yl += err_dt - err_ldt;
    }
    cout << "(author)total diffrence: " << diff_y << " average difference: " << diff_y / nTests << '\n';
    cout << "(wzh)total diffrence: " << diff_w << " average difference: " << diff_w / nTests << '\n';
    cout << "(lq)total diffrence: " << diff_l << " average difference: " << diff_l / nTests << '\n';

    cout << "(wzh)total diffrence: " << diff_yw << " average difference: " << diff_yw / nTests << '\n';
    cout << "(lq)total diffrence: " << diff_yl << " average difference: " << diff_yl / nTests << '\n';
}

int main() {
    testGoICP();
    testdt();
    testFpcsGoicpImprovement();
}

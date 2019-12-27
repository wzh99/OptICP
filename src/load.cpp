#include "load.hpp"

#include <fstream>
#include <boost/make_shared.hpp>
#include <boost/spirit.hpp>

using namespace boost::spirit;

PointCloudPtr LoadPCDFile(const std::string& path, const float samplePercentage) {
    // Open file
    std::ifstream ifs;
    ifs.open(path);
    if (ifs.fail())
        Error("Failed to open PCD file: %s", path.c_str());
    std::string line;

    // Get point number and omit other metadata
    // Since we assume data to be an 1D array, we only need to care about total number.
    auto cloud = boost::make_shared<PointCloudT>();
    auto size = 0u;
    while (true) {
        std::getline(ifs, line);
        auto info = parse(line.c_str(), str_p("POINTS ") >> uint_p[assign_a(size)]);
        if (info.full) {
            cloud->reserve(size);
            break;
        }
    }
    size *= samplePercentage;

    // Jump to where point data begin
    while (true) {
        std::getline(ifs, line);
        auto info = parse(line.c_str(), str_p("DATA"));
        if (info.hit) break;
    }

    std::default_random_engine rng;
    std::uniform_real_distribution<float> distrib;
    // Read actual point data
    while (true) {
        std::getline(ifs, line);
        Point3f p;
        auto info = parse(line.c_str(), real_p[assign_a(p.x)] >> blank_p >> real_p[assign_a(p.y)]
            >> blank_p >> real_p[assign_a(p.z)]);
        if (!info.full)
            break;
        //Error("Wrong format: %s", line.c_str());

    // Simply discard points according to random nambers
    // It is not precise, but accurate enough.
        if (samplePercentage > 0.99 || distrib(rng) < samplePercentage)
            cloud->push_back(p);

        if (cloud->size() == size) break;
    }

    ifs.close();
    return cloud;
}

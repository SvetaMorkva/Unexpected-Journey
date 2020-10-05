#include "ObjectDetectorBrisk.h"

#include <iostream>
#include <map>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::vector<std::string> get_file_list(const std::string& path)
{
    std::vector<std::string> m_file_list;
    if (fs::is_directory(path)) {
        fs::path apk_path(path);
        fs::directory_iterator end;

        for (fs::directory_iterator i(apk_path); i != end; ++i) {
            const fs::path cp = (*i);
            if (cp.extension() == ".jpg" || cp.extension() == ".jpeg") {
                m_file_list.push_back(cp.string());
            }
        }
    }
    return m_file_list;
}

int main() {
    auto algorithm = [](const std::string &type) {
        auto model = cv::imread("../" + type + "_origin.jpg", cv::IMREAD_GRAYSCALE);
        if (model.empty()) {
            std::cerr << "ERROR: Could not process model image!" << std::endl;
            return 1;
        }
        ObjectDetectorBrisk detector(model, "../" + type + "_stat.csv");

        const auto &allFileNames = get_file_list("../img_" + type);

        std::cout << "If you don't want see all this " << allFileNames.size()
                  << " photos, just press esc!\n" << "Otherwise press any other button to see next\n";

        for (size_t i = 0; i < allFileNames.size(); i++) {
            auto train = cv::imread(allFileNames[i], cv::IMREAD_GRAYSCALE);
            float resizeCoef = 0.25f * (i % 5 + 2);
            cv::resize(train, train, cv::Size(), resizeCoef, resizeCoef);

            if (train.empty()) {
                std::cerr << "Warning: Could not train image!" << std::endl;
                continue;
            }

            detector.detectObject(train, allFileNames[i]);
            if (detector.matchDescriptors()) {
                detector.statisticAndDraw();
            }
        }
    };
    algorithm("surf_dataset");
    algorithm("brisk_dataset");
    return 0;
}

#include "ObjectDetector.h"
#include "fileutils.h"

#include <iostream>
#include <map>

std::map<std::string, int> read()
{
    std::ifstream fin;
    std::string line;
    std::map<std::string, int> data;
    // Open an existing file
    fin.open("../metrics/train_data.csv");
    std::getline(fin, line, '\n');
    while(std::getline(fin, line, '\n')) {
        size_t it = line.find(",");
        auto filename = line.substr(0, it);
        auto type = std::stoi(line.substr(it + 1));

        data[filename] = type;
    }
    fin.close();
    return data;
}

void trainClassificator() {
    const auto &trainFileNames = fileutils::get_file_list("../images/train");
    const auto &testFileNames = fileutils::get_file_list("../images/test");
    const auto &intersectFileNames = fileutils::get_file_list("../images/Intersect");
    ObjectDetector detectorBrisk;
    std::vector<cv::Mat> trainImage;
    std::vector<cv::Mat> testImage;
    std::vector<int> labelsTrain, labelsTest;
    auto data = read();
    for (int i = 0; i < trainFileNames.size(); i++) {
        auto train = cv::imread(trainFileNames[i]);

        if (train.empty()) {
            std::cerr << "Warning: Could not train image!" << std::endl;
            continue;
        }
        cvtColor(train, train, cv::COLOR_BGR2GRAY);
        labelsTrain.push_back(data[trainFileNames[i]]);
        trainImage.push_back(train);

    }

    cv::Ptr<cv::ml::SVM> SVMbrisk;
    detectorBrisk.train(trainImage, labelsTrain, SVMbrisk);
//
//    for (int i = 0; i < testFileNames.size(); i++) {
//        auto test = cv::imread(testFileNames[i]);
//
//        if (test.empty()) {
//            std::cerr << "Warning: Could not train image!" << std::endl;
//            continue;
//        }
//        testImage.push_back(test);
//    }
//    for (int i = 0; i < intersectFileNames.size(); i++) {
//        auto test = cv::imread(intersectFileNames[i]);
//
//        if (test.empty()) {
//            std::cerr << "Warning: Could not train image!" << std::endl;
//            continue;
//        }
//        testImage.push_back(test);
//    }
//    detectorBrisk.predict(testImage, SVMbrisk);
    cv::VideoCapture video("../Video/Intersect.mp4");
    detectorBrisk.predictVideo(video, SVMbrisk);

}

int main() {
    fileutils::preprocessData();
    trainClassificator();
    return 0;
}

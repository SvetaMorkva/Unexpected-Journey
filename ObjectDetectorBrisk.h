//
// Created by Sveta Morkva on 10/3/20.
//

#ifndef KPI_LAB2_OBJECTDETECTORBRISK_H
#define KPI_LAB2_OBJECTDETECTORBRISK_H

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

class ObjectDetectorBrisk {
public:
    ObjectDetectorBrisk(const cv::Mat &model, const std::string &filename);
    ~ObjectDetectorBrisk();
    void detectObject(const cv::Mat &train, const std::string &name = "");
    bool matchDescriptors();
    void statisticAndDraw();

private:
    void detectKeyPoints();
    void computeDescriptors();

    struct trainObjectData {
        std::string filename;
        cv::Mat image;
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;
        std::vector<cv::DMatch> matches;
    };

    cv::Mat mModelImage;
    std::vector<cv::KeyPoint> mModelKeyPoints;
    cv::Mat mModelDescriptors;
    trainObjectData mTrain;

    cv::Ptr<cv::BRISK> mBriskDescriptor = nullptr;
    cv::Ptr<cv::DescriptorMatcher> mBfHammingMatcher = nullptr;

    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::ofstream mStatisticFile;

    static bool mDraw;
};


#endif //KPI_LAB2_OBJECTDETECTORBRISK_H

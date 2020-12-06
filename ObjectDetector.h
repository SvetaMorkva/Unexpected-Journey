//
// Created by Sveta Morkva on 10/3/20.
//

#ifndef KPI_LAB2_OBJECTDETECTOR_H
#define KPI_LAB2_OBJECTDETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <future>

#include <fstream>

class ObjectDetector {
public:
    void predict(const std::vector<cv::Mat>& images, cv::Ptr<cv::ml::SVM>& svm);
    void predictVideo(cv::VideoCapture video, cv::Ptr<cv::ml::SVM>& svm);
    void train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm);

private:
    void processFrame(cv::Mat img, cv::Ptr<cv::ml::SVM>& svm);
    std::pair<cv::Mat, std::vector<int>> getDescriptorsForTrain(std::vector<cv::Mat> images, std::vector<int> labels);
    void processModel();
    cv::Mat rowForClassificator(int objNum, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints,
                                std::vector<cv::Point2f> &scene_corners);
    void drawReplace(const std::vector<std::pair<int, std::vector<cv::Point2f>>>& obj_corners);

    std::vector<cv::Mat> mModelDescr, mModelImg, mModelReplaceImg;
    std::vector<std::vector<cv::KeyPoint>> mModelKeyPoints;
    std::vector<std::vector<cv::Point2f>> mObj_corners;
    std::vector<std::vector<cv::Point2f>> mReplaceObj_corners;

    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Mat mCurrentFrame;

    const int num_descr = 15;
    std::map<int, float> ratio_thresh = { {0, 0.707f},
                                          {1, 0.727f} };
    float resizeValue = 0.11;

    const int minimumSize = 230;
};


#endif //KPI_LAB2_OBJECTDETECTOR_H

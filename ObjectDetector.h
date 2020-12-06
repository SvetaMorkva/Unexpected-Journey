//
// Created by Sveta Morkva on 10/3/20.
//

#ifndef KPI_LAB2_OBJECTDETECTOR_H
#define KPI_LAB2_OBJECTDETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

class ObjectDetector {
public:
    ObjectDetector();
    void predict(std::vector<cv::Mat> images, cv::Ptr<cv::ml::SVM>& svm);
    void predictVideo(cv::VideoCapture video, cv::Ptr<cv::ml::SVM>& svm);
    void train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm);

private:
    cv::Mat getDescriptors(std::vector<cv::Mat> images);
    std::pair<cv::Mat, std::vector<int>> getDescriptorsForTrain(std::vector<cv::Mat> images, std::vector<int> labels);
    void processModel();
    bool checkProportionIsRight(std::vector<cv::Point2f> corners, std::vector<cv::Point2f> orgCorners);

    int num_descr = 30;

    std::vector<cv::Mat> mModelDescr, mModelImg, mModelReplaceImg;
    std::vector<std::vector<cv::KeyPoint>> mModelKeyPoints;

    std::map<int, std::vector<cv::Point2f>> mDetectedObjectRect;
};


#endif //KPI_LAB2_OBJECTDETECTOR_H

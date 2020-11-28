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
    ObjectDetectorBrisk(bool brisk);
    void predict(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm);
    void predictVideo(cv::VideoCapture video, cv::Ptr<cv::ml::SVM>& svm);
    void train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm);

private:
    cv::Mat getDescriptors(std::vector<cv::Mat> images);

    int num_clusters = 8;
    int num_descr = 30;
    bool mBriskDescr = true;
    cv::Rect curImgBoundRect;
};


#endif //KPI_LAB2_OBJECTDETECTORBRISK_H

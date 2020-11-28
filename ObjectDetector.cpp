//
// Created by Sveta Morkva on 10/3/20.
//

#include "ObjectDetectorBrisk.h"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::ml;

void ObjectDetectorBrisk::train(std::vector<cv::Mat> images, std::vector<int> labels,
                                cv::Ptr<cv::ml::SVM>& svm) {
    Mat samples = getDescriptors(images);

    svm = SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(samples, ROW_SAMPLE, labels);
    svm->save("../trained_model/trained_svm.xml");
}

void ObjectDetectorBrisk::predict(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm) {

    Mat samples = getDescriptors(images);

    Mat predict_labels;
    svm->predict(samples, predict_labels);
    int truePositives = 0;
    int falsePositives = 0;
    int trueNegatives = 0;
    int falseNegatives = 0;
    for (int i = 0; i < predict_labels.rows; i++) {
        if (predict_labels.at<float>(i, 0) != 0) {
            predict_labels.at<float>(i, 0) == labels[i] ? truePositives++ : falsePositives++;
        } else {
            predict_labels.at<float>(i, 0) == labels[i] ? trueNegatives++ : falseNegatives++;
        }
    }

    double precison = truePositives * 1.0/(truePositives+falsePositives);
    double recall = truePositives * 1.0/(truePositives+falseNegatives);
    double f1 = 2*precison*recall/(precison+recall);
    printf("Precision: %f\nRecall: %f\nF1-score: %f\n", precison, recall, f1);
}

cv::Mat ObjectDetectorBrisk::getDescriptors(std::vector<cv::Mat> images) {

    Ptr<DescriptorExtractor> extractor;
    if (mBriskDescr) {
        extractor = BRISK::create();
    } else {
        extractor = xfeatures2d::SURF::create();
    }

    Mat outSamples = Mat::zeros(images.size(), extractor->descriptorSize() * num_descr, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<int> clustNames;
        std::map<int, std::vector<Point2f>> clusters;
        auto img = images[i];
        extractor->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        if (keypoints.size() < num_clusters) {
            continue;
        }
        cv::Mat points(keypoints.size(), descriptors.cols + 2, CV_32FC1);
        for (int z = 0; z < keypoints.size(); z++) {
            auto p = keypoints[z].pt;
            for (int j = 0; j < descriptors.cols; j++) {
                points.at<float>(z, j) = mBriskDescr ? descriptors.at<uchar>(z, j) : descriptors.at<float>(z, j);
            }
            points.at<float>(z, descriptors.cols) = p.x;
            points.at<float>(z, descriptors.cols + 1) = p.y;
        }
        kmeans(points, num_clusters, clustNames,
               TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
               KMEANS_PP_CENTERS);
        Mat img_keypoints;
        for (int j = 0; j < points.rows; j++) {
            Point2f p = {points.at<float>(j, descriptors.cols),
                         points.at<float>(j, descriptors.cols + 1)};
            clusters[clustNames[j]].push_back(p);
        }

        int biggestClusterNum = 0;
        int biggestClusterName;
        for (auto cl: clusters) {
            if (biggestClusterNum < cl.second.size()*0.95) {
                biggestClusterName = cl.first;
                biggestClusterNum = cl.second.size();
            } else {
                auto r1 = boundingRect(cl.second);
                auto r2 = boundingRect(clusters[biggestClusterName]);
                Point center_of_rect1 = (r1.br() + r1.tl())*0.5;
                Point center_of_rect2 = (r2.br() + r2.tl())*0.5;
                Point center_of_img = {img.cols/2, img.rows/2};
                double dist1 = cv::norm(center_of_img-center_of_rect1);
                double dist2 = cv::norm(center_of_img-center_of_rect2);
                if (dist1 < dist2) {
                    biggestClusterName = cl.first;
                    biggestClusterNum = cl.second.size();
                }
            }

        }
        int num = 0;
        for (int z = 0; z < descriptors.rows; z++) {
            if (clustNames[z] == biggestClusterName) {
                for (int j = 0; j < descriptors.cols; j++) {
                    outSamples.at<float>(i, num*descriptors.cols+j) = points.at<float>(z, j);
                }
                if (++num == num_descr) {
                    break;
                }
            }
        }
        curImgBoundRect = boundingRect(clusters[biggestClusterName]);
//        rectangle(img, r, Scalar(255, 0, 0));
//        imshow("SURF Keypoints", img);
//        waitKey();
    }
    return outSamples;
}

void ObjectDetectorBrisk::predictVideo(cv::VideoCapture video, Ptr<cv::ml::SVM> &svm) {
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) {
            break;
        }
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        auto img = getDescriptors({frame});
        float label = svm->predict(img);
        if (label != 0) {
            rectangle(frame, curImgBoundRect, Scalar(255, 0, 0));
        }

        imshow("Frame", frame);

        char c = (char)waitKey(25);
        if (c == 27) {
            break;
        }
    }

    video.release();
}

ObjectDetectorBrisk::ObjectDetectorBrisk(bool brisk) : mBriskDescr(brisk) {
}


//
// Created by Sveta Morkva on 10/3/20.
//

#include "ObjectDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>
#include <future>

#include "fileutils.h"

using namespace cv;
using namespace cv::ml;

void ObjectDetector::train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm) {

    extractor = BRISK::create();
    matcher = DescriptorMatcher::create(BFMatcher::BRUTEFORCE_HAMMING);

    processModel();
    auto pair = getDescriptorsForTrain(images, labels);

    auto _samples = pair.first;
    auto _labels = pair.second;
    svm = SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10, 1e-6));
    svm->train(_samples, ROW_SAMPLE, _labels);
    svm->save("../model.xml");
}

void ObjectDetector::predict(const std::vector<cv::Mat>& images, cv::Ptr<cv::ml::SVM>& svm) {

    for (const auto &image: images) {
        processFrame(image, svm);
        imshow("Frame", mCurrentFrame);
        waitKey();
    }
}

void ObjectDetector::predictVideo(cv::VideoCapture video, cv::Ptr<cv::ml::SVM>& svm) {
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    int num_frames = 0;
    time_t start, end;
    time(&start);

    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) {
            break;
        }
        num_frames++;
        processFrame(frame, svm);

        imshow("Frame", mCurrentFrame);
        waitKey(1);
    }
    time(&end);
    double seconds = difftime (end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    auto fps  = num_frames / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;
}

void ObjectDetector::processModel() {
    auto origins = fileutils::get_file_list("../origins");
    std::sort(origins.begin(), origins.end());

    for (const auto &name: origins) {
        cv::Mat descriptors;
        std::vector<KeyPoint> keypoints;
        std::vector<Point2f> obj_corners(4), replace_corners(4);
        auto modelImg = cv::imread(name);
        cvtColor(modelImg, modelImg, COLOR_BGR2GRAY);
        resize(modelImg, modelImg, cv::Size(), resizeValue, resizeValue);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f(modelImg.cols, 0);
        obj_corners[2] = Point2f(modelImg.cols, modelImg.rows);
        obj_corners[3] = Point2f(0, modelImg.rows);
        auto modelReplName = "../replace/" + fileutils::getFilenameFromPath(name);
        auto modelReplace = cv::imread(modelReplName);
        replace_corners[0] = Point2f(0, 0);
        replace_corners[1] = Point2f(modelReplace.cols, 0);
        replace_corners[2] = Point2f(modelReplace.cols, modelReplace.rows);
        replace_corners[3] = Point2f(0, modelReplace.rows);
        if (modelImg.empty()) {
            std::cerr << "Warning: Could not train image!" << std::endl;
            return;
        }
        extractor->detect(modelImg, keypoints);
        extractor->compute(modelImg, keypoints, descriptors);
        mModelImg.push_back(modelImg);
        mModelDescr.push_back(descriptors);
        mModelKeyPoints.push_back(keypoints);
        mModelReplaceImg.push_back(modelReplace);
        mObj_corners.push_back(obj_corners);
        mReplaceObj_corners.push_back(replace_corners);
    }
}

std::pair<Mat, std::vector<int>> ObjectDetector::getDescriptorsForTrain(std::vector<cv::Mat> images,
                                                                        std::vector<int> labels) {

    Mat outSamples;
    std::vector<int> _labels;
    outSamples.create(0, extractor->descriptorSize() * num_descr, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        resizeValue = 1.0 * minimumSize / std::min(images[i].cols, images[i].rows);
        resize(images[i], images[i], cv::Size(), resizeValue, resizeValue);
        cvtColor(images[i], images[i], cv::COLOR_BGR2GRAY);

        extractor->detect(images[i], keypoints);
        extractor->compute(images[i], keypoints, descriptors);
        if (keypoints.empty()) {
            continue;
        }
        for (int objNum = 0; objNum < mModelImg.size(); objNum++) {
            std::vector<Point2f> scene_corners(4);
            auto samples = rowForClassificator(objNum, descriptors, keypoints, scene_corners);
            if (!samples.empty()) {
                outSamples.push_back(samples);
                _labels.push_back(labels[i] == objNum + 1 ? labels[i] : 0);
            }
        }
    }
    return {outSamples, _labels};
}

void ObjectDetector::processFrame(cv::Mat img, cv::Ptr<cv::ml::SVM>& svm) {
    std::vector<std::pair<int, std::vector<Point2f>>> obj_corners;
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;

    mCurrentFrame = img;

    resizeValue = 1.0 * minimumSize / std::min(img.cols, img.rows);
    resize(img, img, cv::Size(), resizeValue, resizeValue);
    cvtColor(img, img, COLOR_BGR2GRAY);

    extractor->detect(img, keypoints);
    extractor->compute(img, keypoints, descriptors);
    if (keypoints.empty()) {
        return;
    }

    for (int modelImgNum = 0; modelImgNum < mModelImg.size(); modelImgNum++) {
        std::vector<Point2f> scene_corners(4);
        auto samples = rowForClassificator(modelImgNum, descriptors, keypoints, scene_corners);
        if (!samples.empty() && int(svm->predict(samples)) == modelImgNum + 1) {
            obj_corners.emplace_back(modelImgNum, scene_corners);
        }
    }
    drawReplace(obj_corners);
}

cv::Mat ObjectDetector::rowForClassificator(int objNum, Mat descriptors, std::vector<KeyPoint> keypoints,
                                            std::vector<cv::Point2f> &scene_corners) {
    std::vector<Point2f> obj, scene;
    std::vector<std::vector<DMatch>> knnMatches;
    std::vector<int> goodKeypoints;

    auto modelImg = mModelImg[objNum];
    auto modelDescr = mModelDescr[objNum];
    auto modelKeyPoints = mModelKeyPoints[objNum];
    matcher->knnMatch(modelDescr, descriptors, knnMatches, 2);

    for (size_t m = 0; m < knnMatches.size(); m++) {
        if (knnMatches[m][0].distance < ratio_thresh[objNum] * knnMatches[m][1].distance) {
            obj.push_back(modelKeyPoints[knnMatches[m][0].queryIdx].pt);
            scene.push_back(keypoints[knnMatches[m][0].trainIdx].pt);
            goodKeypoints.push_back(knnMatches[m][0].trainIdx);
        }
    }

    Mat samples = Mat::zeros(1, extractor->descriptorSize() * num_descr, CV_32FC1);

    if (scene.size() < 4) {
        return Mat();
    }
    Mat H = findHomography(obj, scene, RANSAC);

    if (!H.empty()) {
        perspectiveTransform(mObj_corners[objNum], scene_corners, H);
        if (scene_corners.size() != 4) {
            return Mat();
        }
        int num = 0;
        for (auto p: goodKeypoints) {
            if (pointPolygonTest(scene_corners, keypoints[p].pt, false) == 1) {
                for (int z = 0; z < descriptors.cols; z++) {
                    samples.at<float>(0, num * descriptors.cols + z) = descriptors.at<uchar>(p, z);
                }
                if (++num == num_descr) {
                    break;
                }
            }
        }
    }
    return samples;
}

void ObjectDetector::drawReplace(std::vector<std::pair<int, std::vector<Point2f>>> obj_corners) {
    Mat resultImg;
    RNG rng(12345);
    for (const auto &obj: obj_corners) {
        int imgIndex = obj.first;
        auto corners = obj.second;
        for (int i = 0; i < 4; i++) {
            corners[i] = Point2f(corners[i].x / resizeValue, corners[i].y / resizeValue);
        }
        auto persTransform = getPerspectiveTransform(mReplaceObj_corners[imgIndex], corners);
        Mat newImg;
        warpPerspective(mModelReplaceImg[imgIndex], newImg, persTransform,
                        Size(mCurrentFrame.cols, mCurrentFrame.rows));
        if (resultImg.empty()) {
            resultImg = newImg;
        } else {
            for (int i = 0; i < resultImg.rows; i++) {
                for (int j = 0; j < resultImg.cols; j++) {
                    if (resultImg.at<Vec3b>(i, j) != Vec3b(0, 0, 0) &&
                        newImg.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                        resultImg.at<Vec3b>(i, j) = Vec3b(rng.uniform(0, 255),
                                                          rng.uniform(0, 255),
                                                          rng.uniform(0, 255));
                        continue;
                    }
                    resultImg.at<Vec3b>(i, j) += newImg.at<Vec3b>(i, j);
                }
            }
        }
    }
    if (!obj_corners.empty()) {
        for (int i = 0; i < mCurrentFrame.rows; i++) {
            for (int j = 0; j < mCurrentFrame.cols; j++) {
                if (resultImg.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                    mCurrentFrame.at<Vec3b>(i, j) = resultImg.at<Vec3b>(i, j);
                }
            }
        }
    }
}

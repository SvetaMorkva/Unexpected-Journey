//
// Created by Sveta Morkva on 10/3/20.
//

#include "ObjectDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "fileutils.h"

using namespace cv;
using namespace cv::ml;

void ObjectDetector::train(std::vector<cv::Mat> images, std::vector<int> labels, cv::Ptr<cv::ml::SVM>& svm) {
    processModel();
    auto pair = getDescriptorsForTrain(images, labels);

    auto _samples = pair.first;
    auto _labels = pair.second;
    svm = SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10, 1e-6));
    svm->train(_samples, ROW_SAMPLE, _labels);
}

void ObjectDetector::predict(std::vector<cv::Mat> images, cv::Ptr<cv::ml::SVM>& svm) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<cv::Mat> grayImages;
    for (auto image: images) {
        cvtColor(image, image, COLOR_BGR2GRAY);
        grayImages.push_back(image);
    }
    Mat samples = getDescriptors(grayImages);
    for (int i = 0; i < images.size(); i++) {
        imshow("orgImg", images[i]);
        Mat resultImage = images[i];
        for (int objNum = 0; objNum < mModelImg.size(); objNum++) {
            int res = svm->predict(samples.row(i*mModelImg.size()+objNum));
            if (res != 0) {
                std::vector<cv::Point> scene_corners;
                for (const auto &point : mDetectedObjectRect[i*mModelImg.size()+objNum]) {
                    scene_corners.push_back(point);
                }
                polylines(resultImage, scene_corners, true, Scalar(0, 255, 0), 3);

                std::vector<Point2f> obj_corners(4);
                obj_corners[0] = Point2f(0, 0);
                obj_corners[1] = Point2f(mModelReplaceImg[objNum].cols, 0);
                obj_corners[2] = Point2f(mModelReplaceImg[objNum].cols, mModelReplaceImg[objNum].rows);
                obj_corners[3] = Point2f(0, mModelReplaceImg[objNum].rows);
                if (mDetectedObjectRect[i*mModelImg.size()+objNum].size() == 4) {
                    auto persTransform = getPerspectiveTransform(obj_corners,
                                                                 mDetectedObjectRect[i * mModelImg.size() + objNum]);
                    Mat newImg;
                    warpPerspective(mModelReplaceImg[objNum], newImg, persTransform,
                                    Size(resultImage.cols, resultImage.rows));
                    std::vector<std::vector<Point>> co_ordinates;
                    co_ordinates.push_back(scene_corners);
                    Mat mask(resultImage.rows, resultImage.cols, CV_8UC1, cv::Scalar(0));
                    drawContours(mask, co_ordinates, 0, Scalar(255), cv::FILLED, 8);
                    newImg.copyTo(resultImage, mask);
                }
            }
        }

        imshow("Good Matches & Object detection", resultImage);
        waitKey();
    }
}

cv::Mat ObjectDetector::getDescriptors(std::vector<cv::Mat> images) {

    Ptr<DescriptorExtractor> extractor = BRISK::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(BFMatcher::BRUTEFORCE_HAMMING);

    Mat outSamples = Mat::zeros(images.size() * mModelImg.size(),
                                extractor->descriptorSize() * num_descr, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        auto img = images[i];
        extractor->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        if (keypoints.empty()) {
            continue;
        }
        for (int objNum = 0; objNum < mModelImg.size(); objNum++) {
            std::vector<std::vector<DMatch>> knnMatches;
            std::vector<Point2f> obj, scene;

            auto modelImg = mModelImg[objNum];
            auto modelDescr = mModelDescr[objNum];
            auto modelKeyPoints = mModelKeyPoints[objNum];
            matcher->knnMatch(modelDescr, descriptors, knnMatches, 2);

            const float ratio_thresh = 0.7f;
            for (size_t m = 0; m < knnMatches.size(); m++) {
                if (knnMatches[m][0].distance < ratio_thresh * knnMatches[m][1].distance) {
                    obj.push_back(modelKeyPoints[knnMatches[m][0].queryIdx].pt);
                    scene.push_back(keypoints[knnMatches[m][0].trainIdx].pt);
                }
            }

            if (scene.size() < 4) {
                continue;
            }
            Mat H = findHomography(obj, scene, RANSAC);

            std::vector<Point2f> obj_corners(4), scene_corners(4);
            obj_corners[0] = Point2f(0, 0);
            obj_corners[1] = Point2f(modelImg.cols, 0);
            obj_corners[2] = Point2f(modelImg.cols, modelImg.rows);
            obj_corners[3] = Point2f(0, modelImg.rows);
            mDetectedObjectRect[i*mModelImg.size()+objNum] = obj_corners;
            if (!H.empty()) {
                int num = 0;
                perspectiveTransform(obj_corners, scene_corners, H);
                if (scene_corners.size() == 4) {
                    mDetectedObjectRect[i * mModelImg.size() + objNum] = scene_corners;
                }
                for (int j = 0; j < keypoints.size(); j++) {
                    if (pointPolygonTest(scene_corners, keypoints[j].pt, false) != -1) {
                        for (int z = 0; z < descriptors.cols; z++) {
                            outSamples.at<float>(i*mModelImg.size()+objNum, num * descriptors.cols + z) =
                                    descriptors.at<uchar>(j, z);
                        }
                        if (++num == num_descr) {
                            break;
                        }
                    }
                }

            }
        }
    }

    return outSamples;
}

void ObjectDetector::predictVideo(cv::VideoCapture video, cv::Ptr<cv::ml::SVM>& svm) {
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
        Mat resultImage = frame;
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        auto img = getDescriptors({frame});
        for (int i = 0; i < mModelImg.size(); i++) {
            float label = svm->predict(img.row(i));
            if (label != 0) {
                std::vector<cv::Point> scene_corners;
                for (const auto &point : mDetectedObjectRect[i]) {
                    scene_corners.push_back(point);
                }
                polylines(resultImage, scene_corners, true, Scalar(0, 255, 0), 3);

                std::vector<Point2f> obj_corners(4);
                obj_corners[0] = Point2f(0, 0);
                obj_corners[1] = Point2f(mModelReplaceImg[i].cols, 0);
                obj_corners[2] = Point2f(mModelReplaceImg[i].cols, mModelReplaceImg[i].rows);
                obj_corners[3] = Point2f(0, mModelReplaceImg[i].rows);
                if (checkProportionIsRight(mDetectedObjectRect[i], obj_corners)) {
                    auto persTransform = getPerspectiveTransform(obj_corners, mDetectedObjectRect[i]);
                    Mat newImg;
                    warpPerspective(mModelReplaceImg[i], newImg, persTransform,
                                    Size(resultImage.cols, resultImage.rows));
                    std::vector<std::vector<Point>> co_ordinates;
                    co_ordinates.push_back(scene_corners);
                    Mat mask(resultImage.rows, resultImage.cols, CV_8UC1, cv::Scalar(0));
                    drawContours(mask, co_ordinates, 0, Scalar(255), cv::FILLED, 8);
                    newImg.copyTo(resultImage, mask);
                }
            }
        }

        imshow("Frame", resultImage);

        char c = (char)waitKey(25);
        if (c == 27) {
            break;
        }
    }
}

ObjectDetector::ObjectDetector() {
}

std::pair<Mat, std::vector<int>> ObjectDetector::getDescriptorsForTrain(std::vector<cv::Mat> images,
                                                                        std::vector<int> labels) {

    Ptr<DescriptorExtractor> extractor = BRISK::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(BFMatcher::BRUTEFORCE_HAMMING);
    Mat outSamples;
    std::vector<int> _labels;
    outSamples.create(0, extractor->descriptorSize() * num_descr, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;

        extractor->detect(images[i], keypoints);
        extractor->compute(images[i], keypoints, descriptors);
        if (keypoints.empty()) {
            continue;
        }
        for (int objNum = 0; objNum < mModelImg.size(); objNum++) {
            std::vector<Point2f> obj, scene;
            std::vector<std::vector<DMatch>> knnMatches;
            std::vector<int> clustNames;
            auto modelImg = mModelImg[objNum];
            auto modelDescr = mModelDescr[objNum];
            auto modelKeyPoints = mModelKeyPoints[objNum];
            matcher->knnMatch(modelDescr, descriptors, knnMatches, 2);

            const float ratio_thresh = 0.7f;
            for (size_t m = 0; m < knnMatches.size(); m++) {
                if (knnMatches[m][0].distance < ratio_thresh * knnMatches[m][1].distance) {
                    obj.push_back(modelKeyPoints[knnMatches[m][0].queryIdx].pt);
                    scene.push_back(keypoints[knnMatches[m][0].trainIdx].pt);
                }
            }

            if (scene.size() < 4) {
                continue;
            }
            Mat H = findHomography(obj, scene, RANSAC);

            std::vector<Point2f> obj_corners(4), scene_corners(4);
            obj_corners[0] = Point2f(0, 0);
            obj_corners[1] = Point2f(modelImg.cols, 0);
            obj_corners[2] = Point2f(modelImg.cols, modelImg.rows);
            obj_corners[3] = Point2f(0, modelImg.rows);
            if (!H.empty()) {
                int num = 0;
                perspectiveTransform(obj_corners, scene_corners, H);

                Mat samples = Mat::zeros(1, extractor->descriptorSize() * num_descr, CV_32FC1);
                for (int j = 0; j < keypoints.size(); j++) {
                    if (pointPolygonTest(scene_corners, keypoints[j].pt, false) == 1) {
                        for (int z = 0; z < descriptors.cols; z++) {
                            samples.at<float>(0, num * descriptors.cols + z) = descriptors.at<uchar>(j, z);
                        }
                        if (++num == num_descr) {
                            break;
                        }
                    }
                }
                outSamples.push_back(samples);
                if (labels[i] == objNum + 1) {
                    _labels.push_back(labels[i]);
                } else {
                    _labels.push_back(0);
                }
            }
        }
    }
    return {outSamples, _labels};
}

void ObjectDetector::processModel() {
    Ptr<DescriptorExtractor> extractor = BRISK::create();
    auto origins = fileutils::get_file_list("../origins");
    std::sort(origins.begin(), origins.end());

    for (const auto &name: origins) {
        cv::Mat descriptors;
        std::vector<KeyPoint> keypoints;
        auto modelImg = cv::imread(name);
        cvtColor(modelImg, modelImg, COLOR_BGR2GRAY);
        auto modelReplName = "../replace/" + fileutils::getFilenameFromPath(name);
        auto modelReplace = cv::imread(modelReplName);
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
    }
}

bool ObjectDetector::checkProportionIsRight(std::vector<cv::Point2f> corners, std::vector<cv::Point2f> orgCorners) {
    float widthTop = corners[1].x - corners[0].x;
    float widthBottom = corners[2].x - corners[3].x;
    float heightLeft = corners[3].y - corners[0].y;
    float heightRight = corners[2].y - corners[1].y;
    if (widthTop * widthBottom * heightLeft * heightRight < 20) {
        return false;
    }
    return true;
}


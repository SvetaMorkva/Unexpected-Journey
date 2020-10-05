//
// Created by Sveta Morkva on 10/3/20.
//

#include "ObjectDetectorBrisk.h"

using namespace cv;

ObjectDetectorBrisk::ObjectDetectorBrisk(const Mat &model, const std::string &filename) {
    mStatisticFile.open (filename);
    mStatisticFile << "Image_name, Correct features, Localization error, Time (ms), Size\n";
    mBriskDescriptor = BRISK::create();
    mBfHammingMatcher = DescriptorMatcher::create(BFMatcher::BRUTEFORCE_HAMMING);

    detectObject(model);

    mModelImage = mTrain.image;
    mModelKeyPoints = mTrain.keyPoints;
    mModelDescriptors = mTrain.descriptors;
}

void ObjectDetectorBrisk::detectObject(const Mat &train, const std::string &name) {
    mTrain.matches.clear();
    mTrain.image = train;
    mTrain.filename = name;
    detectKeyPoints();
    computeDescriptors();
}

void ObjectDetectorBrisk::detectKeyPoints() {
    mStart = std::chrono::high_resolution_clock::now();
    mBriskDescriptor->detect(mTrain.image, mTrain.keyPoints);
}

void ObjectDetectorBrisk::computeDescriptors() {
    mBriskDescriptor->compute(mTrain.image, mTrain.keyPoints, mTrain.descriptors);
}

bool ObjectDetectorBrisk::matchDescriptors() {
    if (mTrain.descriptors.empty()) {
        return false;
    }

    std::vector<std::vector<DMatch>> knnMatches;
    mBfHammingMatcher->knnMatch(mModelDescriptors, mTrain.descriptors,
                                knnMatches, 2);

    const float ratio_thresh = 0.7f;
    for(size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            mTrain.matches.push_back(knnMatches[i][0]);
        }
    }
    return !mTrain.matches.empty();
}

void ObjectDetectorBrisk::statisticAndDraw() {
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (size_t i = 0; i < mTrain.matches.size(); i++) {
        obj.push_back(mModelKeyPoints[mTrain.matches[i].queryIdx].pt);
        scene.push_back(mTrain.keyPoints[mTrain.matches[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC);

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f(mModelImage.cols, 0);
    obj_corners[2] = Point2f(mModelImage.cols, mModelImage.rows);
    obj_corners[3] = Point2f(0, mModelImage.rows);
    std::vector<Point2f> scene_corners(4);
    if (!H.empty()) {
        perspectiveTransform(obj_corners, scene_corners, H);
        auto end = std::chrono::high_resolution_clock::now();

        std::vector<Point> object_corners;
        for (const auto &point : scene_corners) {
            object_corners.push_back(point);
            object_corners.back().x += mModelImage.cols;
        }

        float point_inside = 0;
        float distance_diff = 0;
        for (size_t i = 0; i < scene.size(); i++) {
            const auto &p = scene[i];
            auto check = pointPolygonTest(scene_corners, p, false);
            if (check == 1 || check == 0) {
                point_inside += 1;
                distance_diff += mTrain.matches[i].distance;
            }
        }
        float pointsInside_avg = point_inside / scene.size();
        float distance_avg = point_inside != 0 ? distance_diff / point_inside : 0;
        float time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-mStart).count();
        mStatisticFile << mTrain.filename << "," << pointsInside_avg << "," << distance_avg << "," <<
            time_diff << "," << mTrain.image.size().width << ":" << mTrain.image.size().height << "\n";

        if (mDraw) {
            Mat result;
            drawMatches(mModelImage, mModelKeyPoints, mTrain.image, mTrain.keyPoints, mTrain.matches, result,
                        Scalar::all(-1), Scalar::all(-1),
                        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            polylines(result, object_corners, true, Scalar(0, 255, 0), 3);

            imshow("Good Matches & Object detection", result);
            char c = (char)waitKey();
            if (c == 27) {
                mDraw = false;
            }
        }
    }
}

ObjectDetectorBrisk::~ObjectDetectorBrisk() {
    mStatisticFile.close();
}

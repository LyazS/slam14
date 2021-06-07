#ifndef __POSE_2D2D_H__
#define __POSE_2D2D_H__

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void pose_estimation_2d2d(
    vector<cv::KeyPoint> kp1,
    vector<cv::KeyPoint> kp2,
    vector<cv::DMatch> matches,
    cv::Mat &E,
    cv::Mat &R,
    cv::Mat &t);
#endif
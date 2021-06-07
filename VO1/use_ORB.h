#ifndef __USE_ORB_H__
#define __USE_ORB_H__
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
    const cv::Mat &img1,
    const cv::Mat &img2,
    vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<cv::DMatch> &matches);

#endif
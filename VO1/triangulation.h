#ifndef __TRIANGULATION_H__
#define __TRIANGULATION_H__

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &k);
void triangulation(
    const vector<cv::KeyPoint> &kp1,
    const vector<cv::KeyPoint> &kp2,
    const vector<cv::DMatch> &matches,
    const cv::Mat &R, const cv::Mat &t,
    vector<cv::Point3d> &tri_points);
#endif
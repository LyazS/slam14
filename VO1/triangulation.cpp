#include "triangulation.h"

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &k)
{
    return cv::Point2f(
        (p.x - k.at<double>(0, 2)) / k.at<double>(0, 0),
        (p.y - k.at<double>(1, 2)) / k.at<double>(1, 1));
}

void triangulation(
    const vector<cv::KeyPoint> &kp1,
    const vector<cv::KeyPoint> &kp2,
    const vector<cv::DMatch> &matches,
    const cv::Mat &R, const cv::Mat &t,
    vector<cv::Point3d> &tri_points)
{
    //相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    //以第一帧为初始化世界坐标
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    //第二帧的相机变换矩阵，经由对极几何求出
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    //匹配点
    vector<cv::Point2d> pt1, pt2;
    for (int i = 0; i < matches.size(); i++)
    {
        pt1.push_back(pixel2cam(kp1[matches[i].queryIdx].pt, K));
        pt2.push_back(pixel2cam(kp2[matches[i].trainIdx].pt, K));
    }

    cv::Mat pt4d;
    cv::triangulatePoints(T1, T2, pt1, pt2, pt4d);

    for (int i = 0; i < pt4d.cols; i++)
    {
        cv::Mat x = pt4d.col(i);
        x /= x.at<double>(3, 0);
        cv::Point3d p(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
        tri_points.push_back(p);
    }
}
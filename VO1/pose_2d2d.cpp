#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "pose_2d2d.h"

using namespace std;
using namespace cv;

void pose_estimation_2d2d(
    vector<cv::KeyPoint> kp1,
    vector<cv::KeyPoint> kp2,
    vector<cv::DMatch> matches,
    cv::Mat &E,
    cv::Mat &R,
    cv::Mat &t)
{
     //相机内参，型号：TUM Freibury2
     cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1,
                  0, 521.0, 249.7,
                  0, 0, 1);

     //将匹配点转换格式
     vector<cv::Point2f> point1, point2;

     for (int i = 0; i < (int)matches.size(); i++)
     {
          point1.push_back(kp1[matches[i].queryIdx].pt);
          point2.push_back(kp2[matches[i].trainIdx].pt);
     }

     //计算基础矩阵
     cv::Mat fundamental_mat;
     fundamental_mat = cv::findFundamentalMat(point1, point2, cv::FM_8POINT);
     cout << "fundamental_mat " << endl
          << fundamental_mat << endl;

     //计算本质矩阵
     //相机光心
     cv::Point2d principal_point(325.1, 249.7);
     //相机焦距
     int focal_length = 521;
     cv::Mat essential_mat;
     essential_mat = cv::findEssentialMat(point1, point2, focal_length, principal_point, cv::RANSAC);
     cout << "essential_mat " << endl
          << essential_mat << endl;

     //计算单应矩阵
     cv::Mat homography_mat;
     homography_mat = cv::findHomography(point1, point2, cv::RANSAC, 3.0, cv::noArray(), 2000, 0.99);
     cout << "homography_mat " << endl
          << homography_mat << endl;

     cv::recoverPose(essential_mat, point1, point2, R, t, focal_length, principal_point);
     cout << "R " << endl
          << R << endl;
     cout << "t " << endl
          << t << endl;
     E = essential_mat;
}
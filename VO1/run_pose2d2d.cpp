#include <iostream>
#include <string>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "pose_2d2d.h"
#include "use_ORB.h"
#include "triangulation.h"
#include "myPCL.h"

int main()
{
    string build_dir = get_current_dir_name();
    string build_str = "build";
    string edit_dir = build_dir.erase(build_dir.length() - build_str.length());
    string img_dir = edit_dir + "testdata/";

    cv::Mat img1 = cv::imread(img_dir + "1.png");
    cv::Mat img2 = cv::imread(img_dir + "2.png");

    vector<cv::KeyPoint> kp1, kp2;
    vector<cv::DMatch> matches;

    //寻找匹配点对
    find_feature_matches(img1, img2, kp1, kp2, matches);
    cout << "matches size: " << matches.size() << endl;

    //估计两张图之间的运动
    cv::Mat R, t, E;
    pose_estimation_2d2d(kp1, kp2, matches, E, R, t);

    //验证一下，E=t^R*scale
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                   t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                   -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cv::Mat t_R = t_x * R;
    cout << "t_R " << t_R << endl;
    cout << "scale = E/t_R = " << endl
         << E / t_R << endl;

    //三角化
    vector<cv::Point3d> tri_points;
    triangulation(kp1, kp2, matches, R, t, tri_points);

    //三角化与重投影关系
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < matches.size(); i++)
    {
        //第一张图
        //图像上的点到相机坐标
        cv::Point2d pt1_cam = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        //三角化计算得到的点位置重投影到相机坐标
        cv::Point2d pt1_cam_3d(tri_points[i].x / tri_points[i].z,
                               tri_points[i].y / tri_points[i].z);
        cout << "point     in cam1 " << pt1_cam << endl;
        cout << "point tri in cam1 " << pt1_cam_3d << " d=" << tri_points[i].z << endl;

        //第二张图
        cv::Point2d pt2_cam = pixel2cam(kp2[matches[i].trainIdx].pt, K);
        cv::Mat pt2_cam_tfs = R * (cv::Mat_<double>(3, 1) << tri_points[i].x, tri_points[i].y, tri_points[i].z) + t;
        pt2_cam_tfs /= pt2_cam_tfs.at<double>(2, 0);
        cout << "point     in cam2 " << pt2_cam << endl;
        cout << "point tri in cam2 " << pt2_cam_tfs.t() << endl;

        break;
    }

    // vector<PointT> pcl_point;
    // for (int i = 0; i < tri_points.size(); i++)
    // {
    //     PointT p;
    //     p.x = tri_points[i].x;
    //     p.y = tri_points[i].y;
    //     p.z = tri_points[i].z;
    //     pcl_point.push_back(p);
    // }
    // SavePointCloud(pcl_point, "test.pcd");
}
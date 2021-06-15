#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <string>
#include <boost/format.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>

#include "myPCL.h"

using namespace std;
using namespace cv;

int mainj()
{
    //相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    vector<cv::Mat> colorImgs, depthImgs;
    vector<Eigen::Isometry3d> poses;

    string build_dir = get_current_dir_name();
    string build_str = "build";
    string edit_dir = build_dir.erase(build_dir.length() - build_str.length());
    string rgbd_dir = edit_dir + "testdata/rgbd/";
    ifstream fin(rgbd_dir + "pose.txt");
    if (!fin)
    {
        cout << "not find poses.txt" << endl;
        return 0;
    }

    for (int i = 0; i < 5; i++)
    {
        boost::format fmt("%s/%d.%s");
        colorImgs.push_back(cv::imread(rgbd_dir + (fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread(rgbd_dir + (fmt % "depth" % (i + 1) % "pgm").str(), -1));

        double data[7] = {0};
        for (int d = 0; d < 7; d++)
        {
            fin >> data[d];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    //计算点云并拼接
    vector<PointT> pointcloud;
    for (int i = 0; i < 5; i++)
    {
        cout << "transform img " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (d == 0)
                    continue;
                //从像素坐标系转换到相机坐标系
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                //从相机坐标系转换到世界坐标系
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                pointcloud.push_back(p);
            }
    }
    SavePointCloud(pointcloud, edit_dir + "test.pcd");
    return 0;
}
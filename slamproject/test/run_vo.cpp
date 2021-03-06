#include <fstream>
#include <string>
#include <boost/timer/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
using namespace cv;
#include "myslam/config.h"
#include "myslam/visual_odometry.h"
using namespace myslam;
int main()
{
    string config_path = "/home/slam14/slamproject/config/default.yaml";
    myslam::Config::SetParameterFile(config_path);
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

    string dataset_dir = myslam::Config::Get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;

    ifstream fin(dataset_dir + "associate.txt");
    if (!fin)
    {
        cout << "not exist associate.txt" << endl;
        return 0;
    }
    //加载视频时间序列
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        rgb_files.push_back(dataset_dir + rgb_file);
        depth_times.push_back(atof(depth_time.c_str()));
        depth_files.push_back(dataset_dir + depth_file);

        if (fin.good() == false)
            break;
    }

    myslam::Camera::Ptr camera(new myslam::Camera);

    //可视化
    cv::viz::Viz3d vis("VO");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    cout << "read total " << rgb_files.size() << " entries" << endl;
    for (int i = 0; i < rgb_files.size(); i++)
    {
        cv::Mat color = cv::imread(rgb_files[i]);
        cv::Mat depth = cv::imread(rgb_files[i], cv::IMREAD_UNCHANGED);
        if (color.data == nullptr || depth.data == nullptr)
        {
            break;
        }

        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        // boost::timer timer;
        vo->addFrame(pFrame);
        // cout << "VO cost time: " << timer.elapsed(). << endl;

        if (vo->state_ == myslam::VisualOdometry::LOST)
            break;

        Sophus::SE3d Tcw = pFrame->T_c_w_.inverse();
        // show the map and the camera pose
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.rotationMatrix()(0, 0), Tcw.rotationMatrix()(0, 1), Tcw.rotationMatrix()(0, 2),
                Tcw.rotationMatrix()(1, 0), Tcw.rotationMatrix()(1, 1), Tcw.rotationMatrix()(1, 2),
                Tcw.rotationMatrix()(2, 0), Tcw.rotationMatrix()(2, 1), Tcw.rotationMatrix()(2, 2)),
            cv::Affine3d::Vec3(
                Tcw.translation()(0, 0), Tcw.translation()(1, 0), Tcw.translation()(2, 0)));

        cv::Mat img_show = color.clone();
        for (auto &pt : vo->map_->map_points_)
        {
            myslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel(p->pos_, pFrame->T_c_w_);
            cv::circle(img_show, cv::Point2f(pixel(0, 0), pixel(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("image", img_show);
        cv::waitKey(1);
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
    }
}
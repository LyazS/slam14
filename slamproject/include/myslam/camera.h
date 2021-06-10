#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "myslam/common_include.h"

namespace myslam
{
    //针孔RGBD相机模型
    class Camera
    {
    public:
        typedef std::shared_ptr<Camera> Ptr;
        //相机内参
        float fx_, fy_, cx_, cy_, depth_scale_;

        Camera();
        Camera(float fx, float fy, float cx, float cy, float depth_scale = 0) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), depth_scale_(depth_scale) {}

        //坐标变换
        Eigen::Vector3d world2cam(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);
        Eigen::Vector3d cam2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w);
        Eigen::Vector3d cam2pixel(const Eigen::Vector3d &p_c);
        Eigen::Vector3d pixel2cam(const Eigen::Vector3d &p_p, double depth = 1);
        Eigen::Vector3d pixel2world(const Eigen::Vector3d &p_p, const Sophus::SE3d &T_c_w, double depth = 1);
        Eigen::Vector3d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);
    };
}
#endif
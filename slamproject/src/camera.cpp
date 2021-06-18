#include "myslam/camera.h"
#include "myslam/config.h"
namespace myslam
{
    Camera::Camera()
    {
        fx_ = Config::Get<float>("camera.fx");
        fy_ = Config::Get<float>("camera.fy");
        cx_ = Config::Get<float>("camera.cx");
        cy_ = Config::Get<float>("camera.cy");
        depth_scale_ = Config::Get<float>("camera.depth_scale");
    }
    Eigen::Vector3d Camera::world2cam(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
    {
        return T_c_w * p_w;
    }
    Eigen::Vector3d Camera::cam2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w)
    {
        return T_c_w.inverse() * p_c;
    }
    Eigen::Vector2d Camera::cam2pixel(const Eigen::Vector3d &p_c)
    {
        Eigen::Vector2d c2p(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_);
        return c2p;
    }
    Eigen::Vector3d Camera::pixel2cam(const Eigen::Vector2d &p_p, double depth)
    {
        Eigen::Vector3d p2c(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth);
        return p2c;
    }
    Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth)
    {
        return cam2world(pixel2cam(p_p, depth), T_c_w);
    }
    Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
    {
        return cam2pixel(world2cam(p_w, T_c_w));
    }
}
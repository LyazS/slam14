#ifndef __FRAME_H__
#define __FRAME_H__
#include "myslam/common_include.h"
#include "myslam/camera.h"
namespace myslam
{
    class Frame
    {
    public:
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_;      //帧id
        double time_stamp_;     //帧时刻
        Sophus::SE3d T_c_w_;    //从世界到相机的坐标转换
        Camera::Ptr camera_;    //相机模型
        cv::Mat color_, depth_; //rgb与深度图

        Frame();
        Frame(long id, double time_stamp = 0, Sophus::SE3d T_c_w = Sophus::SE3d(),
              Camera::Ptr camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());
        ~Frame();

        //工厂函数，创建帧
        static Frame::Ptr createFrame();

        //获取点对应的深度
        double findDepth(const cv::KeyPoint &kp);

        //获取相机光心
        Eigen::Vector3d getCamCenter() const;

        //检查该点是否在该帧里边，是否在视野内
        bool isInFrame(const Eigen::Vector3d &pt_world);
    };

}
#endif
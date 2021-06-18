#ifndef __VISUAL_ODOMETRY_H__
#define __VISUAL_ODOMETRY_H__
#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
    class VisualOdometry
    {
    public:
        typedef std::shared_ptr<VisualOdometry> Ptr;
        enum VOState
        {
            INITIALIZING = -1, //设定第一帧，初始化
            OK = 0,            //顺利跟踪
            LOST               //丢失
        };
        VOState state_;
        Map::Ptr map_;    //包含所有帧和点的地图
        Frame::Ptr ref_;  //参考帧
        Frame::Ptr curr_; //当前帧

        cv::Ptr<cv::ORB> orb_;           //orb检测器和描述计算器
        vector<cv::Point3f> pts_3d_ref_; //参考帧中的3d点
        vector<cv::KeyPoint> kp_curr_;   //当前帧的关键点
        cv::Mat descriptors_curr_;       //当前帧的描述
        cv::Mat descriptors_ref_;        //参考帧的描述
        vector<cv::DMatch> feature_matches_;

        Sophus::SE3d T_c_r_estimated_; //当前帧的估计姿态
        int num_inliners_;             //icp的内点个数
        int num_lost_;                 //丢失次数

        //参数
        int num_of_features_; //特征个数
        double scale_factor_; //特征金字塔缩放尺
        int level_pyramid_;   //特征金字塔层数
        float match_ratio_;   //选择良好匹配的比例
        int max_num_lost_;    //最大连续丢失次数
        int min_inliers_;     //内点最少个数

        double key_frame_min_rot;   //两关键帧之间最小的旋转
        double key_frame_min_trans; //两关键帧之间最小的平移

        bool log=true; //是否打印过程
    public:
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame); //增加新帧

    protected:
        //内部运算
        void extractKeyPoints();
        void computeDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void setRef3DPoints();

        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();
    };
}
#endif
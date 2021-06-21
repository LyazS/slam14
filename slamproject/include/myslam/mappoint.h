#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__
#include "myslam/common_include.h"
#include "frame.h"
namespace myslam
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_;                            //ID
        static unsigned long MapPoint_factory_id; //
        bool good_;                                   //是否是一个好的路标
        Eigen::Vector3d pos_;                         //路标世界坐标
        Eigen::Vector3d norm_;                        //观察角度的方向向量
        cv::Mat descriptor_;                          //特征描述
        list<Frame *> observed_frames_;               //能观察到这个点的关键帧
        int matched_times_;                           //成为内点的次数
        int visible_times_;                           //可见的次数

        int observed_times_; //观测时刻
        int correct_times_;  //

        MapPoint();
        MapPoint(unsigned long id,
                 const Eigen::Vector3d &position,
                 const Eigen::Vector3d &norm,
                 Frame *frame = nullptr,
                 const cv::Mat descriptor = cv::Mat());

        inline cv::Point3f getPositionCV() const
        {
            return cv::Point3f(pos_(0, 0), pos_(1, 0), pos_(2, 0));
        }
        static MapPoint::Ptr createMapPoint();
        static MapPoint::Ptr createMapPoint(
            const Eigen::Vector3d &pos_world,
            const Eigen::Vector3d &norm,
            const cv::Mat &descriptor,
            Frame *frame);
    };
}

#endif
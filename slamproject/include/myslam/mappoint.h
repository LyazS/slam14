#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__
#include "myslam/common_include.h"
namespace myslam
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        Eigen::Vector3d pos_;  //路标世界坐标
        Eigen::Vector3d norm_; //
        cv::Mat descriptor_;   //特征描述
        int observed_times_;   //观测时刻
        int correct_times_;    //

        MapPoint();
        MapPoint(long id, Eigen::Vector3d position, Eigen::Vector3d norm);

        static MapPoint::Ptr createMapPoint();
    };
}

#endif
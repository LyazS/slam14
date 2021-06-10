#ifndef __MAP_H__
#define __MAP_H__
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
namespace myslam
{
    class Map
    {
    public:
        typedef std::shared_ptr<Map> Ptr;
        std::unordered_map<unsigned long, MapPoint::Ptr> map_points_;
        std::unordered_map<unsigned long, Frame::Ptr> keyframes_;

        Map() {}

        void insertKeyFrame(Frame::Ptr frame);
        void insertMapPoint(MapPoint::Ptr map_point);
    };
}
#endif
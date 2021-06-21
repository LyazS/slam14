#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer/timer.hpp>

#include "myslam/visual_odometry.h"
#include "myslam/config.h"
#include "myslam/g2o_types.h"

namespace myslam
{
    VisualOdometry::VisualOdometry() : state_(INITIALIZING), ref_(nullptr), curr_(nullptr),
                                       map_(new Map), num_lost_(0), num_inliners_(0),
                                       matcher_flann_(new cv::flann::LshIndexParams(5, 10, 2))
    {

        num_of_features_ = Config::Get<int>("number_of_features");
        scale_factor_ = Config::Get<double>("scale_factor");
        level_pyramid_ = Config::Get<int>("level_pyramid");
        match_ratio_ = Config::Get<float>("match_ratio");
        max_num_lost_ = Config::Get<float>("max_num_lost");
        min_inliers_ = Config::Get<int>("min_inliers");
        key_frame_min_rot = Config::Get<double>("keyframe_rotation");
        key_frame_min_trans = Config::Get<double>("keyframe_translation");
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }

    VisualOdometry::~VisualOdometry() {}

    bool VisualOdometry::addFrame(Frame::Ptr frame)
    {
        switch (state_)
        {
        case INITIALIZING:
        {
            state_ = OK;
            curr_ = ref_ = frame;
            extractKeyPoints();
            computeDescriptors();
            addKeyFrame(); //第一帧就是个关键帧
            ref_ = curr_;
            break;
        }
        case OK:
        {
            curr_ = frame;
            // curr_->T_c_w_ = ref_->T_c_w_;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            //如果是一个好的预测
            if (checkEstimatedPose() == true)
            {
                //当前位置就更新
                curr_->T_c_w_ = T_c_w_estimated_;
                optimizeMap();
                num_lost_ = 0;
                if (checkKeyFrame() == true)
                {
                    addKeyFrame();
                }
                ref_ = curr_;
            }
            else
            {
                num_lost_++;
                if (num_lost_ > max_num_lost_)
                {
                    state_ = LOST;
                }
                return false;
            }
            break;
        }
        case LOST:
        {
            cout << "vo has lost." << endl;
            break;
        }
        }
        return true;
    }
    //获取关键帧

    //获取当前帧的关键点
    void VisualOdometry::extractKeyPoints()
    {
        orb_->detect(curr_->color_, kp_curr_);
    }
    //计算当前帧的描述子
    void VisualOdometry::computeDescriptors()
    {
        orb_->compute(curr_->color_, kp_curr_, descriptors_curr_);
    }

    //匹配参考帧与当前帧的描述子
    void VisualOdometry::featureMatching()
    {
        vector<cv::DMatch> matches;
        //先找一些地图上已有的点来匹配
        //选择候选点，在视野内的
        cv::Mat desp_map;
        vector<MapPoint::Ptr> candidate;
        for (auto &allpoints : map_->map_points_)
        {
            MapPoint::Ptr &p = allpoints.second;
            //检查是不是在视野内
            if (curr_->isInFrame(p->pos_))
            {
                //加入到候选当中
                p->visible_times_++;
                candidate.push_back(p);
                desp_map.push_back(p->descriptor_);
            }
        }
        matcher_flann_.match(desp_map, descriptors_curr_, matches);

        float min_dis = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2)
                                         { return m1.distance < m2.distance; })
                            ->distance;

        match_3dpts_.clear();
        match_2dkp_index_.clear();
        for (cv::DMatch &m : matches)
        {
            if (m.distance < max<float>(min_dis * match_ratio_, 30.0))
            {
                match_3dpts_.push_back(candidate[m.queryIdx]);
                match_2dkp_index_.push_back(m.trainIdx);
            }
        }
        if (log)
        {
            cout << "good matches: " << match_3dpts_.size() << endl;
        }
    }

    void VisualOdometry::poseEstimationPnP()
    {
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (int index : match_2dkp_index_)
        {
            pts2d.push_back(kp_curr_[index].pt);
        }
        for (MapPoint::Ptr pt : match_3dpts_)
        {
            pts3d.push_back(pt->getPositionCV());
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << curr_->camera_->fx_, 0, curr_->camera_->cx_,
                     0, curr_->camera_->fy_, curr_->camera_->cy_,
                     0, 0, 1);
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliners_ = inliers.rows;
        cout << "PnP inliers: " << num_inliners_ << endl;

        cv::Mat R_rvec;
        cv::Rodrigues(rvec, R_rvec);
        Eigen::Matrix<double, 3, 3> m3d_rvec;
        m3d_rvec << R_rvec.at<double>(0, 0), R_rvec.at<double>(0, 1), R_rvec.at<double>(0, 2),
            R_rvec.at<double>(1, 0), R_rvec.at<double>(1, 1), R_rvec.at<double>(1, 2),
            R_rvec.at<double>(2, 0), R_rvec.at<double>(2, 1), R_rvec.at<double>(2, 2);

        T_c_w_estimated_ = Sophus::SE3d(
            Sophus::SO3d(m3d_rvec),
            Eigen::Vector3d(tvec.at<double>(0, 0),
                            tvec.at<double>(1, 0),
                            tvec.at<double>(2, 0)));

        // 线性方程求解器
        //pose 维度为6，路标维度为2
        auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverPL<6, 2>::PoseMatrixType>>();
        // 矩阵块求解器
        auto block_solver = g2o::make_unique<g2o::BlockSolverPL<6, 2>>(std::move(linear_solver));
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

        //图模型
        g2o::SparseOptimizer optimizer;
        //设置求解器
        optimizer.setAlgorithm(solver);
        //打开调试输出
        optimizer.setVerbose(!log ? true : false);

        //只有一个未知量，就是相机的位姿
        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(
            g2o::SE3Quat(
                T_c_w_estimated_.rotationMatrix(),
                T_c_w_estimated_.translation()));
        optimizer.addVertex(pose);

        for (int i = 0; i < inliers.rows; i++)
        {
            int index = inliers.at<int>(i, 0);

            EdegeProjectXYZ2UVPoseOnly *edge = new EdegeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Eigen::Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setMeasurement(Eigen::Vector2d(pts2d[index].x, pts2d[index].y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            // set the inlier map points
            match_3dpts_[index]->matched_times_++;
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_w_estimated_ = Sophus::SE3d(
            pose->estimate().rotation(),
            pose->estimate().translation());
    }

    bool VisualOdometry::checkEstimatedPose()
    {
        //如果内点个数比预设的数量要少，则认为是一帧差的帧
        if (num_inliners_ < min_inliers_)
        {
            if (log)
            {
                cout << "reject because inlier is too small " << num_inliners_ << endl;
            }
            return false;
        }
        auto T_r_c = T_c_w_estimated_ * ref_->T_c_w_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        //如果平移距离过大，则可能匹配变差，由此也认为是差的帧
        if (d.norm() > 5.0)
        {
            if (log)
            {
                cout << "reject because motion is too large: " << d.norm() << endl;
            }
            return false;
        }
        return true;
    }

    //只有两帧之间的旋转与平移都大于预设最小值，才认为是关键帧
    bool VisualOdometry::checkKeyFrame()
    {
        auto T_r_c = T_c_w_estimated_ * ref_->T_c_w_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans)
            return true;
        return false;
    }
    //添加关键帧入map中
    void VisualOdometry::addKeyFrame()
    {

        //对于第一帧，将所有的关键点都加进来地图里
        if (map_->keyframes_.empty())
        {
            for (size_t i = 0; i < kp_curr_.size(); i++)
            {
                double d = curr_->findDepth(kp_curr_[i]);
                if (d < 0)
                    continue;
                Eigen::Vector3d p_world = curr_->camera_->pixel2world(
                    Eigen::Vector2d(
                        kp_curr_[i].pt.x, kp_curr_[i].pt.y),
                    curr_->T_c_w_, d);
                Eigen::Vector3d n = p_world - curr_->getCamCenter();
                n.normalize();
                MapPoint::Ptr map_point =
                    MapPoint::createMapPoint(p_world,
                                             n,
                                             descriptors_curr_.row(i).clone(),
                                             curr_.get());
                map_->insertMapPoint(map_point);
            }
        }
        map_->insertKeyFrame(curr_);
    }
    void VisualOdometry::optimizeMap()
    {
        //移除几乎看不到的点和看不见的点
        for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();)
        {
            //不在视野内的，移除
            if (!curr_->isInFrame(iter->second->pos_))
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            //匹配次数过少的，移除
            float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
            if (match_ratio < map_point_erase_ratio_)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            //观测角度过大的，移除
            double angle = getViewAngle(curr_, iter->second);
            if (angle > M_PI / 6.)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            //如果这个路标点不够好，就用三角化计算该点位置
            if (iter->second->good_ == false)
            {
                //TODO
            }
            iter++;
        }

        //相较于高博的，这句if很关键。
        //多了就基本不可能进入加点的func，地图会越来越小直到没有
        //因此我把它注释掉了
        // if (match_2dkp_index_.size() < 100)
        addMapPoints();

        //如果地图太大，就移除一些点
        if (map_->map_points_.size() > 1000)
        {
            map_point_erase_ratio_ += 0.05;
        }
        else
        {
            map_point_erase_ratio_ = 0.1;
        }

        if (log)
        {
            cout << "Map Size: " << map_->map_points_.size() << endl;
        }
    }

    //计算观测该点的相机方向与该相机方向的角度
    double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
    {
        Eigen::Vector3d n = point->pos_ - frame->getCamCenter();
        n.normalize();
        return acos(n.transpose() * point->norm_);
    }

    //增加地图点
    void VisualOdometry::addMapPoints()
    {
        vector<bool> matched(kp_curr_.size(), false);
        for (int index : match_2dkp_index_)
        {
            matched[index] = true;
        }

        for (int i = 0; i < kp_curr_.size(); i++)
        {
            //这些点已经与地图中的点匹配了，
            //说明这些点已经存在于地图，无需再重复添加进来
            if (matched[i] == true)
                continue;
            double d = curr_->findDepth(kp_curr_[i]);
            if (d < 0)
                continue;
            Eigen::Vector3d p_world = curr_->camera_->pixel2world(
                Eigen::Vector2d(kp_curr_[i].pt.x, kp_curr_[i].pt.y),
                curr_->T_c_w_, d);
            Eigen::Vector3d n = p_world - curr_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get());

            map_->insertMapPoint(map_point);
        }
    }
}
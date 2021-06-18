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
    VisualOdometry::VisualOdometry() : state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliners_(0)
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
            map_->insertKeyFrame(frame);
            extractKeyPoints();
            computeDescriptors();
            //初始化参考帧
            setRef3DPoints();
            break;
        }
        case OK:
        {
            curr_ = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            //如果是一个好的预测
            if (checkEstimatedPose() == true)
            {
                //当前位置就更新
                curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;
                ref_ = curr_;
                setRef3DPoints();
                num_lost_ = 0;
                if (checkKeyFrame() == true)
                {
                    addKeyFrame();
                }
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
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(descriptors_ref_, descriptors_curr_, matches);

        float min_dis = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2)
                                         { return m1.distance < m2.distance; })
                            ->distance;

        feature_matches_.clear();
        for (cv::DMatch &m : matches)
        {
            if (m.distance < max<float>(min_dis * match_ratio_, 30.0))
            {
                feature_matches_.push_back(m);
            }
        }
        if (log)
        {
            cout << "good matches: " << feature_matches_.size() << endl;
        }
    }

    //将当前帧的信息转移到参考帧中
    void VisualOdometry::setRef3DPoints()
    {
        pts_3d_ref_.clear();
        descriptors_ref_ = cv::Mat();
        for (size_t i = 0; i < kp_curr_.size(); i++)
        {
            double d = ref_->findDepth(kp_curr_[i]);
            if (d > 0)
            {
                Eigen::Vector3d p_cam = ref_->camera_->pixel2cam(
                    Eigen::Vector2d(kp_curr_[i].pt.x, kp_curr_[i].pt.y), d);
                pts_3d_ref_.push_back(
                    cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
                descriptors_ref_.push_back(descriptors_curr_.row(i));
            }
        }
    }

    void VisualOdometry::poseEstimationPnP()
    {
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (cv::DMatch m : feature_matches_)
        {
            pts3d.push_back(pts_3d_ref_[m.queryIdx]);
            pts2d.push_back(kp_curr_[m.trainIdx].pt);
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_,
                     0, ref_->camera_->fy_, ref_->camera_->cy_,
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

        T_c_r_estimated_ = Sophus::SE3d(
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
        optimizer.setVerbose(log ? true : false);

        //只有一个未知量，就是相机的位姿
        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(
            g2o::SE3Quat(
                T_c_r_estimated_.rotationMatrix(),
                T_c_r_estimated_.translation()));
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
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_r_estimated_ = Sophus::SE3d(
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
        Sophus::Vector6d d = T_c_r_estimated_.log();
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
        Sophus::Vector6d d = T_c_r_estimated_.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans)
            return true;
        return false;
    }
    //添加关键帧入map中
    void VisualOdometry::addKeyFrame()
    {
        if (log)
        {
            cout << "adding a key-frame" << endl;
        }
        map_->insertKeyFrame(curr_);
    }
}
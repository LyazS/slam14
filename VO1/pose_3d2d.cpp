#include "pose_3d2d.h"

void bundleAdjustment(
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat &K,
    Mat &R, Mat &t)
{
    // 构建图优化，先设定g2o
    // typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;                                  // 每个误差项优化变量维度为6，误差值维度为3
    // Block::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    // Block *solver_ptr = new Block(linearSolver);                                                   // 矩阵块求解器
    // // 梯度下降方法，从GN, LM, DogLeg 中选
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 线性方程求解器
    //pose 维度为6，路标维度为3
    auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
    // 矩阵块求解器
    auto block_solver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    //图模型
    g2o::SparseOptimizer optimizer;
    //设置求解器
    optimizer.setAlgorithm(solver);
    //打开调试输出
    optimizer.setVerbose(true);

    //add vertex
    //cam pose
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
        R_mat,
        Eigen::Vector3d(
            t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);

    //添加路标
    int index = 1;
    for (const Point3f p : points_3d)
    {
        g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();
        point->setId(index);
        index += 1;
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //添加相机参数
    g2o::CameraParameters *camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //添加边
    index = 1;
    for (const Point2f p : points_2d)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(100);
    cout << "T= " << endl
         << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}
#include <iostream>
#include <vector>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
using namespace std;

//曲线模型的顶点，模板参数：优化变量的维度与数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

//误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double _x;
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    //计算曲线模型误差
    virtual void computeError() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement -
                       std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    //计算雅可比矩阵
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;
    vector<double> x_data, y_data;

    cout << "创建数据集" << endl;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        cout << x_data[i] << " " << y_data[i] << endl;
    }

    // 构建图优化，先设定g2o
    // //每个误差项优化变量维度为3，误差值维度为1
    // typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;                                // 每个误差项优化变量维度为3，误差值维度为1
    // Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
    // Block *solver_ptr = new Block(linearSolver);                                      // 矩阵块求解器
    // // 梯度下降方法，从GN, LM, DogLeg 中选
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //pose 维度为6，路标维度为3
    auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverTraits<3, 1>::PoseMatrixType>>();
    // 矩阵块求解器
    auto block_solver = g2o::make_unique<g2o::BlockSolverTraits<3, 1>>(std::move(linear_solver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    //图模型
    g2o::SparseOptimizer optimizer;
    //设置求解器
    optimizer.setAlgorithm(solver);
    //打开调试输出
    optimizer.setVerbose(true);

    //在图里添加顶点
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    //添加边
    for (int i = 0; i < N; i++)
    {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);           //设置其连接的顶点
        edge->setMeasurement(y_data[i]); //观测数值
        //设置信息矩阵：协方差矩阵之逆
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }

    cout << "optimization start" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    //输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "abc_estimate" << abc_estimate.transpose() << endl;
    return 0;
}
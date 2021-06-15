// #define FMT_HEADER_ONLY
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
using namespace std;
using namespace Eigen;

int mainus()
{
    //构造李群
    //沿z轴旋转90度
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Eigen::Quaterniond q(R);
    Sophus::SO3d SO3_R(R); //从R构造
    Sophus::SO3d SO3_q(q); //从q构造

    cout << "SO3_R: " << SO3_R.matrix() << endl;
    cout << "SO3_q: " << SO3_q.matrix() << endl;

    //用对数映射构造李代数
    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3: " << so3.transpose() << endl;

    //hat为向量到反对称矩阵
    Eigen::Matrix3d so3_hat = Sophus::SO3d::hat(so3);
    cout << "so3_hat: " << so3_hat << endl;

    //vee为反对称矩阵到向量
    Eigen::Vector3d so3_vee = Sophus::SO3d::vee(so3_hat);
    cout << "so3_vee: " << so3_vee.transpose() << endl;

    //增量扰动模型的更新
    Eigen::Vector3d update_so3(1e-4, 0, 0);                      //李代数
    Sophus::SO3d update_so3_exp = Sophus::SO3d::exp(update_so3); //指数映射为李群
    cout << "update_so3_exp: \n"
         << update_so3_exp.matrix() << endl;
    Sophus::SO3d SO3_updated = update_so3_exp * SO3_R; //李群上的微小扰动
    cout << "SO3_updated \n"
         << SO3_updated.matrix() << endl;

    cout << "*******************************************" << endl;
    Eigen::Vector3d t(1, 0, 0);
    Sophus::SE3d SE3_Rt(R, t);
    Sophus::SE3d SE3_qt(q, t);
    cout << "SE3_Rt: \n"
         << SE3_Rt.matrix() << endl;
    cout << "SE3_qt: \n"
         << SE3_qt.matrix() << endl;

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3: \n"
         << se3.transpose() << endl;

    //hat为向量到反对称矩阵
    Eigen::Matrix4d se3_hat = Sophus::SE3d::hat(se3);
    cout << "se3_hat: " << se3_hat << endl;

    //vee为反对称矩阵到向量
    Vector6d se3_vee = Sophus::SE3d::vee(se3_hat);
    cout << "se3_vee: " << se3_vee.transpose() << endl;

    //更新
    Vector6d updated_se3;
    updated_se3.setZero();
    updated_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated_exp = Sophus::SE3d::exp(updated_se3);
    cout << "SE3_updated_exp: \n"
         << SE3_updated_exp.matrix() << endl;

    return 0;
}
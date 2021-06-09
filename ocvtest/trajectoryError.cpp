#include <iostream>
#include <fstream>
#include <unistd.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>
using namespace Sophus;
using namespace std;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
TrajectoryType ReadTrajectory(const string &path);
TrajectoryType ReadTrajectory(const string &path)
{
    ifstream fin(path);
    TrajectoryType trajectory;
    if (!fin)
    {
        cout << "not read " << path << endl;
        return trajectory;
    }
    while (!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d p1(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz));

        trajectory.push_back(p1);
    }
    return trajectory;
}

int main3()
{
    string gt_path = "../testdata/trajectory/groundtruth.txt";
    string es_path = "../testdata/trajectory/estimated.txt";
    TrajectoryType gt = ReadTrajectory(gt_path);
    TrajectoryType es = ReadTrajectory(es_path);
    cout << "gt size: " << gt.size() << " es size: " << es.size() << endl;

    //计算RMSE
    double rmse = 0;
    for (size_t i = 0; i < es.size(); i++)
    {
        Sophus::SE3d p1 = es[i], p2 = gt[i];
        Sophus::SE3d p2invp1 = p2.inverse() * p1;
        if (i == 0)
            cout << p2invp1.matrix() << endl;
        double error = p2invp1.log().norm();
        rmse += error * error;
    }
    rmse /= double(es.size());
    rmse = sqrt(rmse);
    cout << "RMSE: " << rmse << endl;
    return 0;
}
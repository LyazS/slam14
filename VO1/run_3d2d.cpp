#include "pose_3d2d.h"
#include <unistd.h>
int main()
{
    string build_dir = get_current_dir_name();
    string build_str = "build";
    string edit_dir = build_dir.erase(build_dir.length() - build_str.length());
    string img_dir = edit_dir + "testdata/";

    cv::Mat img1 = cv::imread(img_dir + "1.png",cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(img_dir + "2.png",cv::IMREAD_COLOR);
    cv::Mat d1 = cv::imread(img_dir + "1_depth.png", cv::IMREAD_UNCHANGED);

    vector<cv::KeyPoint> kp1, kp2;
    vector<cv::DMatch> matches;
    find_feature_matches(img1, img2, kp1, kp2, matches);
    cout << "find_feature_matches: " << matches.size() << endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m : matches)
    {
        ushort d = d1.ptr<unsigned short>(int(kp1[m.queryIdx].pt.y))[int(kp1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(kp1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(kp2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    cv::Mat r, t;
    //用opencv的pnp方法求解
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    cv::Mat R;
    //r为旋转向量模式，转换为旋转矩阵
    cv::Rodrigues(r, R);
    cout << "R " << R << endl;
    cout << "t " << t << endl;
}
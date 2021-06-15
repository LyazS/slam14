#include <iostream>
#include <string>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "use_ORB.h"

using namespace std;
using namespace cv;

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2,
                          vector<cv::DMatch> &good_matches)
{
    //初始化
    cv::Mat desc1, desc2;
    Ptr<ORB> orb = cv::ORB::create(
        500, 1.2f, 8, 32, 0, 2,
        ORB::HARRIS_SCORE, 31, 20);
    
    //第一步：检测Oriented FAST角点位置
    orb->detect(img1, kp1);
    orb->detect(img2, kp2);

    //第二步：根据角点位置计算BRIEF描述子
    orb->compute(img1, kp1, desc1);
    orb->compute(img2, kp2, desc2);

    // cout << "描述子 size: " << desc1.size() << endl;

    //第三步：使用Hamming距离，匹配两幅图像中的BRIEF描述子
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    //第四步：筛选匹配点对
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < desc1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    cout << "Max dist: " << max_dist << " "
         << "Min dist: " << min_dist << endl;

    //当描述子距离大于最小值的两倍时，则判断为误匹配
    //当然，以上纯属经验之谈，实际最小值应该设置下限值
    for (int i = 0; i < desc1.rows; i++)
    {
        if (matches[i].distance <= cv::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //第五步：绘制结果
    // cv::Mat outimg1;
    // cv::drawKeypoints(img1, kp1, outimg1, Scalar::all(-1),
    //                   cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("ORB feature point", outimg1);

    // cv::Mat img_match, img_goodmatch;
    // cv::drawMatches(img1, kp1, img2, kp2, matches, img_match);
    // cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_goodmatch);
    // cv::imshow("all match", img_match);
    // cv::imshow("good match", img_goodmatch);
    // cv::waitKey(0);
}

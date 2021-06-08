#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "optical_flow.h"
using namespace std;
using namespace cv;

int main()
{
    string path_to_dataset = "../testdata/data/";
    string associate_file = path_to_dataset + "associate.txt";

    ifstream fin(associate_file);
    if (!fin)
    {
        cerr << "I cann't find associate.txt!" << endl;
        return 1;
    }

    string rgb_file, depth_file, time_rgb, time_depth;
    list<cv::KeyPoint> keypoints; // 因为要删除跟踪失败的点，使用list
    cv::Mat now_color, last_color;

    for (int index = 0; index < 100; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        now_color = cv::imread(path_to_dataset + rgb_file);
        if (now_color.data == nullptr)
            continue;

        if (index == 0)
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(now_color, kps);
            for (auto kp : kps)
                keypoints.push_back(kp);
            last_color = now_color;
            continue;
        }

        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> prev_keypoints;
        vector<cv::Point2f> next_keypoints;
        for (auto kp : keypoints)
            prev_keypoints.push_back(kp.pt);
        // opencv LK
        vector<unsigned char> status;
        vector<float> error;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(last_color, now_color, prev_keypoints, next_keypoints, status, error);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "LK Flow use time：" << time_used.count() << " seconds." << endl;

        //slam14 牛顿法 单层光流
        vector<cv::KeyPoint> kp1_mylk;
        for (auto kp : keypoints)
            kp1_mylk.push_back(kp);
        vector<cv::KeyPoint> kp2_single;
        vector<bool> success_single;
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(last_color, now_color, kp1_mylk, kp2_single, success_single, false, false);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "my LK Flow single use time：" << time_used.count() << " seconds." << endl;

        //slam14 牛顿法 多层光流
        vector<cv::KeyPoint> kp2_multi;
        vector<bool> success_multi;
        t1 = chrono::steady_clock::now();
        OpticalFlowMultiLevel(last_color, now_color, kp1_mylk, kp2_multi, success_multi, true);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "my LK Flow multi use time：" << time_used.count() << " seconds." << endl;

        // 把跟丢的点删掉
        int i = 0;
        cout << status.size() << " " << success_single.size() << " " << success_multi.size() << endl;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); i++)
        {

            // if (status[i] == 0)
            if (success_multi[i] == false)
            {
                //利用erase删除当前指针内容的同时，获取list下一个位置的指针
                iter = keypoints.erase(iter);
                continue;
            }
            //更新当前kp的点坐标
            // (*iter).pt = next_keypoints[i];
            *iter = kp2_multi[i];
            iter++;
        }
        cout << "tracked keypoints: " << keypoints.size() << endl;
        if (keypoints.size() == 0)
        {
            cout << "all keypoints are lost." << endl;
            break;
        }
        // 画出 keypoints
        cv::Mat img_show = now_color.clone();
        for (auto kp : keypoints)
            cv::circle(img_show, kp.pt, 10, cv::Scalar(0, 240, 0), 1);
        cv::imshow("corners", img_show);
        cv::waitKey(0);

        last_color = now_color;
    }
    return 0;
}
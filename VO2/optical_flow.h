#ifndef __OPTICAL_FLOW_H__
#define __OPTICAL_FLOW_H__
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
class OpticalFlowTracer
{
private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const vector<cv::KeyPoint> &kp1;
    vector<cv::KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse;
    bool has_init;

public:
    OpticalFlowTracer(const cv::Mat &img1_, const cv::Mat &img2_,
                      const vector<cv::KeyPoint> &kp1_, vector<cv::KeyPoint> &kp2_,
                      vector<bool> &success_, bool inverse_ = true, bool has_init_ = false) : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_init(has_init_) {}

    void calculateOF(const cv::Range &range);
};
void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false, bool has_init = false);
void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false);
#endif
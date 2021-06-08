#include "optical_flow.h"
#include <Eigen/Core>
#include <Eigen/Dense>
/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols)
        x = img.cols - 1;
    if (y >= img.rows)
        y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_init)
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracer oft(img1, img2, kp1, kp2, success, inverse, has_init);
    //并行化
    cv::parallel_for_(cv::Range(0, kp1.size()), std::bind(&OpticalFlowTracer::calculateOF, &oft, placeholders::_1));
}

void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse)
{
    //金字塔参数
    int pyramids = 4;
    double pyramids_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    //创建金字塔
    vector<cv::Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramids_scale,
                                pyr1[i - 1].rows * pyramids_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramids_scale,
                                pyr2[i - 1].rows * pyramids_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    //多层光流
    vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    //初始化点位置
    for (auto &kp : kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--)
    {
        //从粗糙到精细
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);

        if (level > 0)
        {
            for (auto &kp : kp1_pyr)
                kp.pt /= pyramids_scale;
            for (auto &kp : kp2_pyr)
                kp.pt /= pyramids_scale;
        }
    }
    for (auto &kp : kp2_pyr)
        kp2.push_back(kp);
}

void OpticalFlowTracer::calculateOF(const cv::Range &range)
{
    int half_patch_size = 4;
    int iterations = 10;

    for (size_t i = range.start; i < range.end; i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        //估计dxdy
        if (has_init)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; //假设该点成功追踪到

        //hessian矩阵
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        //bias
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        //Jacobian
        Eigen::Vector2d J;
        for (int iter = 0; iter < iterations; iter++)
        {
            if (inverse == false)
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else
            {
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;

            //计算损失以及Jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {
                    double error = GetPixelValue(img1,
                                                 kp.pt.x + x,
                                                 kp.pt.y + y) -
                                   GetPixelValue(img2,
                                                 kp.pt.x + x + dx,
                                                 kp.pt.y + y + dy);
                    if (inverse == false)
                    {
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img2,
                                                            kp.pt.x + x + dx + 1,
                                                            kp.pt.y + y + dy) -
                                              GetPixelValue(img2,
                                                            kp.pt.x + x + dx - 1,
                                                            kp.pt.y + y + dy)),
                                       0.5 * (GetPixelValue(img2,
                                                            kp.pt.x + x + dx,
                                                            kp.pt.y + y + dy + 1) -
                                              GetPixelValue(img2,
                                                            kp.pt.x + x + dx,
                                                            kp.pt.y + y + dy - 1)));
                    }
                    else if (iter == 0)
                    {
                        //如果inverse模式，即计算一次，对于所有迭代就计一野
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img1,
                                                            kp.pt.x + x + dx + 1,
                                                            kp.pt.y + y + dy) -
                                              GetPixelValue(img1,
                                                            kp.pt.x + x + dx - 1,
                                                            kp.pt.y + y + dy)),
                                       0.5 * (GetPixelValue(img1,
                                                            kp.pt.x + x + dx,
                                                            kp.pt.y + y + dy + 1) -
                                              GetPixelValue(img1,
                                                            kp.pt.x + x + dx,
                                                            kp.pt.y + y + dy - 1)));
                    }
                    b += error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0)
                    {
                        H += J * J.transpose();
                    }

                    //update
                    Eigen::Vector2d update = H.ldlt().solve(b);
                    if (std::isnan(update[0]))
                    {
                        cout << "patch 无梯度信息" << endl;
                        succ = false;
                        break;
                    }
                    if (iter > 0 && cost > lastCost)
                    {
                        //???啥意思这个
                        break;
                    }
                    //update dxdy
                    dx += update[0];
                    dy += update[1];
                    lastCost = cost;
                    succ = true;
                    if (update.norm() < 1e-2)
                    {
                        //收敛了
                        break;
                    }
                }
            success[i] = succ;
            kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
        }
    }
}
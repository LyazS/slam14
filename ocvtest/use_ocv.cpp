#include <iostream>
#include <unistd.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int mainuo()
{
	cout << "程序运行当前目录" << get_current_dir_name() << endl;
	string image_path = "../testdata/useocv/1.png";
	cv::Mat image;
	image = cv::imread(image_path);
	if (image.data == nullptr)
	{
		cout << "not read file" << endl;
		return 0;
	}
	cout << "read ok" << endl;
	cv::imshow("image", image);
	cv::waitKey(0);
	cout << "img type: " << image.type() << endl;

	//count time
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
		{
			//行的头指针
			unsigned char *row_ptr = image.ptr<unsigned char>(y);
			//指向单个像素
			unsigned char *data_ptr = &row_ptr[x * image.channels()];
			//指向通道
			for (int c = 0; c != image.channels(); c++)
			{
				unsigned char data = data_ptr[c];
			}
		}
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "遍历用时 " << time_used.count() << endl;

	//mat 浅复制
	cv::Mat img_a = image;
	img_a(cv::Rect(100, 100, 300, 300)).setTo(0);
	cv::imshow("img", image);
	cv::waitKey(0);

	//mat 深复制
	cv::Mat img_b = image.clone();
	img_b(cv::Rect(100, 100, 300, 300)).setTo(0);
	cv::imshow("img", image);
	cv::imshow("imgb", img_b);
	cv::waitKey(0);
	return 0;
}
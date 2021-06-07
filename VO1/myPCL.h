#ifndef __MYPCL__H__
#define __MYPCL__H__
#include <vector>
#include <string>
using namespace std;
class PointT
{
public:
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    unsigned char r = 255;
    unsigned char g = 255;
    unsigned char b = 255;
};
bool SavePointCloud(vector<PointT> vpt, string file_name);

#endif
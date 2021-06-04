#ifndef __MYPCL__H__
#define __MYPCL__H__
#include <vector>
#include <string>
using namespace std;
class PointT
{
public:
    float x;
    float y;
    float z;
    unsigned char r;
    unsigned char g;
    unsigned char b;
};
bool SavePointCloud(vector<PointT> vpt, string file_name);

#endif
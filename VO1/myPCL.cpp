#include "myPCL.h"
#include <iostream>
#include <fstream>

using namespace std;

bool SavePointCloud(vector<PointT> vpt, string file_name)
{
    int len_pc = vpt.size();

    cout << "points num " << len_pc << "\n";
    fstream f;
    f.open(file_name, ios::out);
    f << "# .PCD v0.7 - Point Cloud Data file format"
      << "\n";
    f << "VERSION 0.7"
      << "\n";
    f << "FIELDS x y z rgb"
      << "\n";
    f << "SIZE 4 4 4 4"
      << "\n";
    f << "TYPE F F F U"
      << "\n";
    f << "COUNT 1 1 1 1"
      << "\n";
    f << "WIDTH " << len_pc << "\n";
    f << "HEIGHT 1"
      << "\n";
    f << "VIEWPOINT 0 0 0 1 0 0 0"
      << "\n";
    f << "POINTS " << len_pc << "\n";
    f << "DATA ascii"
      << "\n";
    for (int i = 0; i < vpt.size(); i++)
    {
        PointT pt = vpt[i];
        int rgb = ((int)pt.r << 16 | (int)pt.g << 8 | (int)pt.b);
        // float frgb = *reinterpret_cast<float*>(&rgb); 
        f << pt.x << " " << pt.y << " " << pt.z << " " << rgb << "\n";
    }
    f.close();
    return true;
}
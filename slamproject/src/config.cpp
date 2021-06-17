#include "myslam/config.h"

namespace myslam
{
    void Config::SetParameterFile(const std::string &filename)
    {
        if (config_ == nullptr)
            config_ = std::shared_ptr<Config>(new Config);

        config_->file_.open(filename, cv::FileStorage::READ);
        if (config_->file_.isOpened() == false)
        {
            cout << "parameter file " << filename << " does not exist." << endl;
            config_->file_.release();
            return;
        }
        else
        {
            cout << "parameter file " << filename << " open." << endl;
        }
    }
    Config::~Config()
    {
        if (file_.isOpened())
            file_.release();
    }
    std::shared_ptr<Config> Config::config_ = nullptr;
}
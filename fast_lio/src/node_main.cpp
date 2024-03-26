#include <Eigen/Core>

#include <so3_math.h>
#include <common_lib.h>
#include "IMU_Processing.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "laserMapping.h"

#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    ROS_INFO("Let's Start FAST-LIO.");
    // fast_lio::FastLIO fast_lio_mapping(nh);
    std::shared_ptr<fast_lio::FastLIO> fast_lio_instance = fast_lio::FastLIO::getInstance(nh);
    ros::spin();

    return 0;
}

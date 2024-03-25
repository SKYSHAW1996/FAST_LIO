// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#ifndef LASER_MAPPING_NODE_H
#define LASER_MAPPING_NODE_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>

#include <Eigen/Core>

#include <so3_math.h>
#include <common_lib.h>
#include "IMU_Processing.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree_impl.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

using namespace ikdtreeNS;
template class KD_TREE<pcl::PointXYZ>;

namespace fast_lio{

class FastLIO 
{
public:
    static std::shared_ptr<FastLIO> getFastLIO(ros::NodeHandle nh);
    static std::shared_ptr<FastLIO> getFastLIO();
    ~FastLIO();

private:
    FastLIO(ros::NodeHandle nh);
    FastLIO(const FastLIO &);
    FastLIO &operator=(const FastLIO &);

    static std::mutex FastLIO_Mutex;
    static shared_ptr<FastLIO> FastLIO_instance;

public:
    condition_variable sig_buffer;
    std::shared_ptr<KD_TREE<PointType>> ikdtree = std::make_shared<KD_TREE<PointType>>();
    vector<PointVector>  Nearest_Points;    //每个点的最近点序列

    PointCloudXYZI::Ptr feats_down_body;    //畸变纠正后降采样的单帧点云，lidar系
    PointCloudXYZI::Ptr feats_down_world;   //畸变纠正后降采样的单帧点云，w系
    PointCloudXYZI::Ptr normvec;            //特征点在地图中对应点的，局部平面参数,w系
    PointCloudXYZI::Ptr laserCloudOri;      // laserCloudOri是畸变纠正后降采样的单帧点云，body系
    PointCloudXYZI::Ptr corr_normvect;      //对应点法相量

    // /*** Time Log Variables ***/
    int time_log_counter = 0;
    int scan_count = 0;
    int publish_count = 0;
    double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
    double solve_const_H_time = 0;
    int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
    bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false;
    bool path_en = true;
    // /**************************/

    std::vector<float> res_last;            //残差，点到面距离平方和
    std::vector<bool> point_selected_surf;  // 是否为平面特征点

    bool flg_exit = false;
    double res_mean_last = 0.05;
    double total_residual = 0.0;
    int effct_feat_num = 0;
    int feats_down_size = 0;
    double solve_time = 0;
    bool extrinsic_est_en = true;

public:
    mutex mtx_buffer;

    const float MOV_THRESHOLD = 1.0f;
    float DET_RANGE = 300.0f;

    double time_diff_lidar_to_imu = 0.0;
    double first_lidar_time = 0.0;
    double lidar_end_time = 0;
    double last_timestamp_lidar = 0;
    double last_timestamp_imu = -1.0;

    // IMU Params
    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double b_gyr_cov = 0.0001;
    double b_acc_cov = 0.0001;

    // LiDAR Preprocess DS Params  
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;  
    double filter_size_surf_min = 0;
    double filter_size_map_min = 0;
    double lidar_mean_scantime = 0.0;
    int    scan_num = 0;

    // // Preprocess & IMU Process pointer
    shared_ptr<Preprocess> p_pre = nullptr; // 定义指向激光雷达数据的预处理类Preprocess的智能指针
    shared_ptr<ImuProcess> p_imu = nullptr; // 定义指向IMU数据预处理类ImuProcess的智能指针

    // // iKDtree
    V3F XAxisPoint_body{LIDAR_SP_LEN, 0.0, 0.0};    //雷达相对于body系的X轴方向的点
    V3F XAxisPoint_world{LIDAR_SP_LEN, 0.0, 0.0};   //雷达相对于world系的X轴方向的点
    BoxPointType LocalMap_Points;       // ikd-tree中,局部地图的包围盒角点
    bool Localmap_Initialized = false;  // 局部地图是否初始化
    double cube_len = 0;                // ikd-tree，局部地图的设置立方体长度
    vector<BoxPointType> cub_needrm;    // ikd-tree中，地图需要移除的包围盒序列

    // /*** EKF inputs and output ***/
    int NUM_MAX_ITERATIONS = 0;
    MeasureGroup Measures;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;    // 状态，噪声维度，输入
    state_ikfom state_point;                            // 状态
    vect3 pos_lid;                                      // world系下lidar坐标

    /*** Some Data Containers ***/
    deque<double>                     time_buffer;
    deque<PointCloudXYZI::Ptr>        lidar_buffer;
    deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    

    // Params for pcd saving
    int keyframe_pulse_cout = 0; 
    int pcd_save_interval = -1;
    int pcd_index = 0;

    bool   lidar_pushed = false, flg_first_scan = true, flg_EKF_inited = false;
    bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

    // // NOT USED PARAMS
    double filter_size_corner_min = 0; // not used
    double fov_deg = 0; // not used
    double FOV_DEG = 0; // not used
    double HALF_FOV_COS = 0; // not used
    double total_distance = 0; // not used
    int laserCloudValidNum = 0; // not used
    int iterCount = 0; // not used
    vector<vector<int>>  pointSearchInd_surf; 

    std::string root_dir = ROOT_DIR;
    std::string map_file_path, lid_topic, imu_topic;

    std::vector<double> extrinT;
    std::vector<double> extrinR;
    V3D euler_cur;
    V3D position_last{Zero3d};
    V3D Lidar_T_wrt_IMU{Zero3d};
    M3D Lidar_R_wrt_IMU{Eye3d};

    PointCloudXYZI::Ptr featsFromMap;       //提取地图中的特征点，IKD-tree获得
    PointCloudXYZI::Ptr feats_undistort;    //去畸变的特征
    PointCloudXYZI::Ptr _featsArray;        // ikd-tree中，map需要移除的点云序列，用于可视化

    PointCloudXYZI::Ptr pcl_wait_pub;
    PointCloudXYZI::Ptr pcl_wait_save;

    nav_msgs::Path path;
    nav_msgs::Odometry odomAftMapped;
    geometry_msgs::Quaternion geoQuat;
    geometry_msgs::PoseStamped msg_body_pose;

    ros::NodeHandle nh_;
    ros::Timer sync_packages_timer_;

    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    ros::Publisher pubLaserCloudFull;
    ros::Publisher pubLaserCloudFull_body;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;

public:
    template<typename T>
    void set_posestamp(T & out);
    
    void publish_odometry(const ros::Publisher & pubOdomAftMapped);
    void publish_frame_world(const ros::Publisher & pubLaserCloudFull);
    void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body);
    void publish_effect_world(const ros::Publisher & pubLaserCloudEffect);
    void publish_map(const ros::Publisher & pubLaserCloudMap);
    void publish_path(const ros::Publisher pubPath);

    void save_pcd();

private:
    void process_data_packages(const ::ros::TimerEvent& timer_event);

    void dump_lio_state_to_log(FILE *fp);

    // Transformation Methods
    template<typename T>
    void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po);
    void pointBodyToWorld(PointType const * const pi, PointType * const po);
    //把点从body系转到world系，通过ikfom的位置和姿态
    void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s);
    // 含有RGB的点云从body系转到world系
    void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
    void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);

    // iKDTree Methods
    void points_cache_collect();    // 通过ikdtree，得到被剔除的点
    void lasermap_fov_segment();    // 在拿到eskf前馈结果后，动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
    void map_incremental();         // 地图的增量更新，主要完成对ikd-tree的地图建立

    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
    bool sync_packages(MeasureGroup &meas);

};

//按下ctrl+c后唤醒所有线程
void SigHandle(int sig)
{
    std::shared_ptr<FastLIO> fast_lio_instance = FastLIO::getFastLIO();
    fast_lio_instance->flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    // 会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞。
    fast_lio_instance->sig_buffer.notify_all();

    fast_lio_instance->save_pcd();
}

//计算残差信息
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    std::shared_ptr<FastLIO> fast_lio_instance = FastLIO::getFastLIO();

    if(fast_lio_instance->ikdtree == nullptr || fast_lio_instance->feats_down_body == nullptr || 
        fast_lio_instance->feats_down_world == nullptr ||  fast_lio_instance->normvec == nullptr || 
        fast_lio_instance->laserCloudOri == nullptr || fast_lio_instance->corr_normvect == nullptr) return;
    
    fast_lio_instance->laserCloudOri->clear();  //将body系的有效点云存储清空
    fast_lio_instance->corr_normvect->clear();  //将对应的法向量清空
    fast_lio_instance->total_residual = 0.0;    //將残差值归零

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    //对降采样后的每个特征点进行残差计算
    for (int i = 0; i < fast_lio_instance->feats_down_size; i++)
    {
        PointType &point_body  = fast_lio_instance->feats_down_body->points[i]; 
        PointType &point_world = fast_lio_instance->feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = fast_lio_instance->Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            fast_lio_instance->ikdtree->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，则认为该点不是有效点
            fast_lio_instance->point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!fast_lio_instance->point_selected_surf[i]) continue;

        VF(4) pabcd;                                        //平面方程参数信息
        fast_lio_instance->point_selected_surf[i] = false;  //将该点设置为无效点，用来计算是否为平面点
        // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            //如果残差大于阈值，则认为该点是有效点
            if (s > 0.9)
            {
                fast_lio_instance->point_selected_surf[i] = true;       //再次回复为有效点
                fast_lio_instance->normvec->points[i].x = pabcd(0);     //将法向量存储至normvec
                fast_lio_instance->normvec->points[i].y = pabcd(1);
                fast_lio_instance->normvec->points[i].z = pabcd(2);
                fast_lio_instance->normvec->points[i].intensity = pd2;  //将点到平面的距离(残差量),存储至normvec的intensit中
                fast_lio_instance->res_last[i] = abs(pd2);              //将残差存储至res_last
            }
        }
    }
    
    //有效特征点数
    fast_lio_instance->effct_feat_num = 0;
    for (int i = 0; i < fast_lio_instance->feats_down_size; i++)
    {
        if (fast_lio_instance->point_selected_surf[i])
        {   
            // body点存到laserCloudOri中, 拟合平面点存到corr_normvect中
            fast_lio_instance->laserCloudOri->points[fast_lio_instance->effct_feat_num] = fast_lio_instance->feats_down_body->points[i];
            fast_lio_instance->corr_normvect->points[fast_lio_instance->effct_feat_num] = fast_lio_instance->normvec->points[i];
            fast_lio_instance->total_residual += fast_lio_instance->res_last[i];
            fast_lio_instance->effct_feat_num ++;
        }
    }

    if (fast_lio_instance->effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    fast_lio_instance->res_mean_last = fast_lio_instance->total_residual / fast_lio_instance->effct_feat_num;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 测量雅可比矩阵H和测量向量的计算 H=J*P*J'
    ekfom_data.h_x = MatrixXd::Zero(fast_lio_instance->effct_feat_num, 12); //测量雅可比矩阵H，论文中的23
    ekfom_data.h.resize(fast_lio_instance->effct_feat_num);                 //测量向量h

    // 求观测值与误差的雅克比矩阵，如论文式14以及式12、13
    for (int i = 0; i < fast_lio_instance->effct_feat_num; i++)
    {
        const PointType &laser_p  = fast_lio_instance->laserCloudOri->points[i];
        // 在body系下，计算点的反对称矩阵, 从点值转换到叉乘矩阵
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        // 转换到IMU坐标系下，计算点的反对称矩阵, 从点值转换到叉乘矩阵
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        // 得到对应的曲面/角的法向量
        const PointType &norm_p = fast_lio_instance->corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        // 计算测量雅可比矩阵H，见fatlio v1的论文公式(14)，求导这部分和LINS相同：https://zhuanlan.zhihu.com/p/258972164
        /*** 
        FAST-LIO2的特别之处
		1.在IESKF中，状态更新可以看成是一个优化问题，即对位姿状态先验 x_bk_bk+1 的偏差，以及基于观测模型引入的残差函数f  的优化问题。
		2.LINS的特别之处在于，将LOAM的后端优化放在了IESKF的更新过程中实现，也就是用IESKF的迭代更新过程代替了LOAM的高斯牛顿法
        ***/
        V3D C(s.rot.conjugate() *norm_vec);         // R^-1 * 法向量,  s.rot.conjugate（）是四元数共轭，即旋转求逆
        V3D A(point_crossmat * C);                  // imu坐标系的点坐标的反对称点乘C
        if (fast_lio_instance->extrinsic_est_en)
        {
            //带be的是激光雷达原始坐标系的点云，不带be的是imu坐标系的点坐标, 优化外参变化矩阵
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); 
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        // 提供点到面的距离
        ekfom_data.h(i) = -norm_p.intensity;
    }
}



} // end of namespace fast_lio

#endif  // LASER_MAPPING_NODE_H
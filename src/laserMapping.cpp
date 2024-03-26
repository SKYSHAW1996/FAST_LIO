// #include <so3_math.h>
// #include "preprocess.h"
// #include <ikd-Tree/ikd_Tree.h>

#include "laserMapping.h"
#include <functional>

namespace fast_lio{

std::mutex FastLIO::mtx_inistance_;
shared_ptr<FastLIO> FastLIO::fastlio_instance_ = nullptr;

std::shared_ptr<FastLIO> FastLIO::getInstance(ros::NodeHandle nh) {
    // std::cout << "Let's get a FastLIO instance." << std::endl;
    if (fastlio_instance_ == nullptr) {
        std::unique_lock<std::mutex> lock(mtx_inistance_);
        if (fastlio_instance_ == nullptr) {
            auto temp = std::shared_ptr<FastLIO>(new FastLIO(nh));
            // std::cout << "Let's get a FastLIO instance with param handler." << std::endl;
            fastlio_instance_ = temp;
        }
    }
    return fastlio_instance_;
}

std::shared_ptr<FastLIO> FastLIO::getInstance() {
    // std::cout << "try to get a FastLIO instance without roshandler." << std::endl;
    if (fastlio_instance_ == nullptr) {
        std::unique_lock<std::mutex> lock(mtx_inistance_);
        if (fastlio_instance_ == nullptr) {
            ros::NodeHandle nh;
            auto temp = std::shared_ptr<FastLIO>(new FastLIO(nh));
            // std::cout << "Let's get a FastLIO instance with tmp handler." << std::endl;
            fastlio_instance_ = temp;
        }
    }
    return fastlio_instance_;
}

FastLIO::FastLIO(ros::NodeHandle nh) : nh_(nh), p_pre_(std::make_shared<Preprocess>()), p_imu_(std::make_shared<ImuProcess>())
{
    // std::vector<float> res_last_tmp(100000, 0.0);
    // std::vector<bool> point_selected_surf_tmp(100000, true);
    res_last_.clear();
    point_selected_surf_.clear();
    res_last_.resize(100000, 0.0);
    point_selected_surf_.resize(100000, true);

    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
    normvec_.reset(new PointCloudXYZI(100000, 1));
    laser_cloud_ori_.reset(new PointCloudXYZI(100000, 1));
    corr_normvect_.reset(new PointCloudXYZI(100000, 1));

    nh_.param<bool>("publish/path_en_",path_en_, true);
    nh_.param<bool>("publish/scan_publish_en",scan_pub_en_, true);
    nh_.param<bool>("publish/dense_publish_en",dense_pub_en_, true);
    nh_.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en_, true);
    nh_.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh_.param<string>("map_file_path",map_file_path,"");
    nh_.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh_.param<string>("common/imu_topic", imu_topic,"/imu/data");
    nh_.param<bool>("common/time_sync_en_", time_sync_en_, false);
    nh_.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu_, 0.0);
    
    nh_.param<double>("filter_size_surf",filter_size_surf_min_,0.5);
    nh_.param<double>("filter_size_map",filter_size_map_min_,0.5);
    nh_.param<double>("cube_side_length",localmap_cube_len_,200);
    nh_.param<float>("mapping/det_range",det_range_,300.f);
    nh_.param<double>("mapping/fov_degree",fov_deg,180);
    nh_.param<double>("mapping/gyr_cov",gyr_cov_,0.1);
    nh_.param<double>("mapping/acc_cov",acc_cov_,0.1);
    nh_.param<double>("mapping/b_gyr_cov",b_gyr_cov_,0.0001);
    nh_.param<double>("mapping/b_acc_cov",b_acc_cov_,0.0001);
    nh_.param<double>("preprocess/blind", p_pre_->blind, 0.01);
    nh_.param<double>("preprocess/max_range", p_pre_->max_range, 150);
    nh_.param<int>("preprocess/lidar_type", p_pre_->lidar_type, AVIA);
    nh_.param<int>("preprocess/scan_line", p_pre_->N_SCANS, 16);
    nh_.param<int>("preprocess/timestamp_unit", p_pre_->time_unit, US);
    nh_.param<int>("preprocess/scan_rate", p_pre_->SCAN_RATE, 10);
    nh_.param<int>("point_filter_num", p_pre_->point_filter_num, 2);
    nh_.param<bool>("feature_extract_enable", p_pre_->feature_enabled, false);
    nh_.param<bool>("runtime_pos_log_enable", runtime_pos_log_, 0);
    nh_.param<bool>("mapping/extrinsic_est_en_", extrinsic_est_en_, true);
    nh_.param<bool>("pcd_save/pcd_save_en_", pcd_save_en_, false);
    nh_.param<int>("pcd_save/interval", pcd_save_interval_, -1);

    nh_.param<vector<double>>("mapping/extrinsic_T", extrinT_, vector<double>());
    nh_.param<vector<double>>("mapping/extrinsic_R", extrinR_, vector<double>());
    cout<<"p_pre_->lidar_type "<<p_pre_->lidar_type<<endl;
    cout<<"cube_side_length "<<localmap_cube_len_<<endl;
    
    path_.header.stamp    = ros::Time::now();
    path_.header.frame_id ="camera_init";

    feats_from_map_.reset(new PointCloudXYZI());
    feats_undistort_.reset(new PointCloudXYZI());

    pcl_wait_pub_.reset(new PointCloudXYZI());
    pcl_wait_save_.reset(new PointCloudXYZI());
    feats_array_.reset(new PointCloudXYZI());

    // memset(point_selected_surf_, true, sizeof(point_selected_surf_));
    // memset(res_last_, -1000.0f, sizeof(res_last_));

    ds_filter_surf_.setLeafSize(filter_size_surf_min_, filter_size_surf_min_, filter_size_surf_min_);
    ds_filter_map_.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);

    Lidar_T_wrt_IMU_<<VEC_FROM_ARRAY(extrinT_);
    Lidar_R_wrt_IMU_<<MAT_FROM_ARRAY(extrinR_);
    p_imu_->set_extrinsic(Lidar_T_wrt_IMU_, Lidar_R_wrt_IMU_);
    p_imu_->set_gyr_cov(V3D(gyr_cov_, gyr_cov_, gyr_cov_));
    p_imu_->set_acc_cov(V3D(acc_cov_, acc_cov_, acc_cov_));
    p_imu_->set_gyr_bias_cov(V3D(b_gyr_cov_, b_gyr_cov_, b_gyr_cov_));
    p_imu_->set_acc_bias_cov(V3D(b_acc_cov_, b_acc_cov_, b_acc_cov_));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf_.init_dyn_share(get_f, df_dx, df_dw, H_Share_Model, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    sub_pcl_ = nh_.subscribe(lid_topic, 200000, &FastLIO::standardPclCallback, this);
    sub_imu_ = nh_.subscribe(imu_topic, 200000, &FastLIO::imuCallback, this);

    cloud_full_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1);
    body_cloud_full_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 1);
    cloud_effect_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 1);
    map_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/Odometry", 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/path", 1);

//------------------------------------------------------------------------------------------------------
    sync_packages_timer_ = nh_.createTimer(::ros::Duration(0.005), &FastLIO::processDataPackages, this);
    signal(SIGINT, SigHandle);
}

FastLIO::~FastLIO() {}

void FastLIO::processDataPackages(const ::ros::TimerEvent& timer_event)
{
    if(syncPackages(data_measures_)) 
        {
            if (first_scan_flg_)
            {
                first_lidar_time_ = data_measures_.lidar_beg_time;
                p_imu_->first_lidar_time_ = first_lidar_time_;
                first_scan_flg_ = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            kdtree_search_time_ = 0.0;
            solve_time_ = 0;
            solve_const_H_time_ = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu_->Process(data_measures_, kf_, feats_undistort_);
            state_point_ = kf_.get_x();
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

            if (feats_undistort_->empty() || (feats_undistort_ == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                return;
            }

            ekf_inited_flg_ = (data_measures_.lidar_beg_time - first_lidar_time_) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            mapFovUpdate();

            /*** downsample the feature points in a scan ***/
            ds_filter_surf_.setInputCloud(feats_undistort_);
            ds_filter_surf_.filter(*feats_down_body_);
            t1 = omp_get_wtime();
            feats_down_size_ = feats_down_body_->points.size();

            /*** initialize the map kdtree ***/
            if(ikdtree_->Root_Node == nullptr)
            {
                if(feats_down_size_ > 5)
                {
                    ikdtree_->set_downsample_param(filter_size_map_min_);
                    feats_down_world_->resize(feats_down_size_);
                    for(int i = 0; i < feats_down_size_; i++)
                    {
                        pointBodyToWorld(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
                    }
                    ikdtree_->Build(feats_down_world_->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree_->validnum();
            kdtree_size_st_ = ikdtree_->size();
            // cout<<"[ mapping ]: In num: "<<feats_undistort_->points.size()<<" downsamp "<<feats_down_size_<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num_<<endl;
            // cout<<"kdtree_size_st_ "<<kdtree_size_st_<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size_ < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                return;
            }
            
            normvec_->resize(feats_down_size_);
            feats_down_world_->resize(feats_down_size_);

            V3D ext_euler = SO3ToEuler(state_point_.offset_R_L_I);

            if(1) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree_->PCL_Storage);
                ikdtree_->flatten(ikdtree_->Root_Node, ikdtree_->PCL_Storage, NOT_RECORD);
                feats_from_map_->clear();
                feats_from_map_->points = ikdtree_->PCL_Storage;
            }

            // pointSearchInd_surf_.resize(feats_down_size_);
            nearest_points_.resize(feats_down_size_);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
            geo_quat_.x = state_point_.rot.coeffs()[0];
            geo_quat_.y = state_point_.rot.coeffs()[1];
            geo_quat_.z = state_point_.rot.coeffs()[2];
            geo_quat_.w = state_point_.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publishOdometry(odom_pub_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            mapIncrement();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en_)                         publishPath(path_pub_);
            if (scan_pub_en_ || pcd_save_en_) {
                if(++keyframe_pulse_cout_ > 10){
                    publishFrameWorld(cloud_full_pub_);
                    keyframe_pulse_cout_ = 0;
                }
            }
            if (scan_pub_en_ && scan_body_pub_en_) publishFrameBody(body_cloud_full_pub_);
            // publishEffectWorld(cloud_effect_pub_);
            publishMap(map_cloud_pub_);

        }
}

void FastLIO::savePcd()
{
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save_->size() > 0 && pcd_save_en_)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }
}

void FastLIO::dumpStateToLog(FILE *fp)  
{
    V3D rot_ang(Log(state_point_.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", data_measures_.lidar_beg_time - first_lidar_time_);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point_.pos(0), state_point_.pos(1), state_point_.pos(2));    // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega  
    fprintf(fp, "%lf %lf %lf ", state_point_.vel(0), state_point_.vel(1), state_point_.vel(2));    // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point_.bg(0), state_point_.bg(1), state_point_.bg(2));       // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point_.ba(0), state_point_.ba(1), state_point_.ba(2));       // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point_.grav[0], state_point_.grav[1], state_point_.grav[2]); // gravity  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void FastLIO::pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void FastLIO::pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I*p_body + state_point_.offset_T_L_I) + state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void FastLIO::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I*p_body + state_point_.offset_T_L_I) + state_point_.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void FastLIO::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I*p_body + state_point_.offset_T_L_I) + state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void FastLIO::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point_.offset_R_L_I*p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void FastLIO::pointsCacheCollect()
{
    PointVector points_history;
    ikdtree_->acquire_removed_points(points_history);
    feats_array_->clear();
    for (int i = 0; i < points_history.size(); i++) feats_array_->push_back(points_history[i]);
    // cout<<">>>>>>> kdtree acquire_removed_points size: "<< feats_array_->size() << endl;

}

void FastLIO::mapFovUpdate()
{   
    // 清空上一次需要移除的区域的数据量
    localmap_cub_remove_.clear(); 
    kdtree_delete_counter_ = 0;
    kdtree_delete_time_ = 0.0;    

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // not used
    // 获取 world 系下 lidar 位置
    V3D pos_LiD = pos_lidar_;
    if (!localmap_Initialized_flg_){
        for (int i = 0; i < 3; i++){
            localmap_boxpoints_.vertex_min[i] = pos_LiD(i) - localmap_cube_len_ / 2.0;
            localmap_boxpoints_.vertex_max[i] = pos_LiD(i) + localmap_cube_len_ / 2.0;
        }
        localmap_Initialized_flg_ = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 获取当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - localmap_boxpoints_.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - localmap_boxpoints_.vertex_max[i]);
        // 与某个方向上的边界距离（例如1.0*100m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= mov_thresh_ * det_range_ || dist_to_map_edge[i][1] <= mov_thresh_ * det_range_) need_move = true;
    }
    if (!need_move) return;

    // 计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = localmap_boxpoints_;
    float mov_dist = max((localmap_cube_len_ - 2.0 * mov_thresh_ * det_range_) * 0.5 * 0.9, double(det_range_ * (mov_thresh_ -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = localmap_boxpoints_;
        if (dist_to_map_edge[i][0] <= mov_thresh_ * det_range_){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = localmap_boxpoints_.vertex_max[i] - mov_dist;
            localmap_cub_remove_.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= mov_thresh_ * det_range_){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = localmap_boxpoints_.vertex_min[i] + mov_dist;
            localmap_cub_remove_.push_back(tmp_boxpoints);
        }
    }
    localmap_boxpoints_ = New_LocalMap_Points;

    // pointsCacheCollect();
    double delete_begin = omp_get_wtime();
    if(localmap_cub_remove_.size() > 0) {
        kdtree_delete_counter_ = ikdtree_->Delete_Point_Boxes(localmap_cub_remove_);
        // std::cout<<">>>>>>> kdtree_delete_counter_: "<< kdtree_delete_counter_ << std::endl;
    }
    kdtree_delete_time_ = omp_get_wtime() - delete_begin;
}

void FastLIO::mapIncrement()
{
    PointVector PointToAdd;                 //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample;      //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size_);
    PointNoNeedDownsample.reserve(feats_down_size_);
    //根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size_; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
        /* decide if need add to map */
        if (!nearest_points_[i].empty() && ekf_inited_flg_)
        {
            const PointVector &points_near = nearest_points_[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            
            // mid_point即为该特征点所属的栅格的中心点坐标
            PointType downsample_result, mid_point; 
            // filter_size_map_min是地图体素降采样的栅格边长
            mid_point.x = floor(feats_down_world_->points[i].x/filter_size_map_min_)*filter_size_map_min_ + 0.5 * filter_size_map_min_;
            mid_point.y = floor(feats_down_world_->points[i].y/filter_size_map_min_)*filter_size_map_min_ + 0.5 * filter_size_map_min_;
            mid_point.z = floor(feats_down_world_->points[i].z/filter_size_map_min_)*filter_size_map_min_ + 0.5 * filter_size_map_min_;
            // 当前点与box中心的距离
            float dist  = calc_dist(feats_down_world_->points[i],mid_point);
            // 判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min_ && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min_ && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min_){
                // 若三个方向距离都大于地图栅格半轴长，无需降采样
                PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                // 如果存在邻近点到中心的距离小于当前点到中心的距离，则不需要添加当前点
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world_->points[i]);
        }
        else
        {
            //如果周围没有点或者没有初始化EKF，则加入到PointToAdd中
            PointToAdd.push_back(feats_down_world_->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    int add_point_size_1 = ikdtree_->Add_Points(PointToAdd, true);
    int add_point_size_2 = ikdtree_->Add_Points(PointNoNeedDownsample, false); 
    add_point_size_ = add_point_size_1 + add_point_size_2;
    // std::cout << "kdtree_incremental add_point_size_: " << add_point_size_ << std::endl;
    kdtree_incremental_time_ = omp_get_wtime() - st_time;
}

void FastLIO::standardPclCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer_.lock();
    scan_count_ ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar_)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer_.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre_->process(msg, ptr);
    lidar_buffer_.push_back(ptr);
    time_buffer_.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar_ = msg->header.stamp.toSec();

    mtx_buffer_.unlock();
    sig_buffer_.notify_all();
}

void FastLIO::imuCallback(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count_ ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu_);
    if (abs(time_diff_lidar_to_imu_) > 0.1 && time_sync_en_)
    {
        msg->header.stamp = \
        ros::Time().fromSec(time_diff_lidar_to_imu_ + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer_.lock();

    if (timestamp < last_timestamp_imu_)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;

    imu_buffer_.push_back(msg);
    mtx_buffer_.unlock();
    sig_buffer_.notify_all();
}

bool FastLIO::syncPackages(MeasureGroup &meas)
{
    // 此处的sync限制了发布的频率，因为当lidar_buffer为空时，sync_packages返回为false，并不会利用新的imu数据进行predict来发布位姿的递推值
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    // ROS_INFO(">>>>>>>>>> lidar_buffer_ size: %d", lidar_buffer_.size());
    // ROS_INFO(">>>>>>>>>> imu_buffer_ size: %d", imu_buffer_.size());

    /*** push a lidar scan ***/
    if(!lidar_pushed_flg_)
    {
        meas.lidar = lidar_buffer_.front();
        meas.lidar_beg_time = time_buffer_.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_)
        {
            lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
        }
        else
        {
            scan_num_ ++;
            lidar_end_time_ = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime_ += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        meas.lidar_end_time_ = lidar_end_time_;

        lidar_pushed_flg_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_)
    {
        // ROS_WARN("last_timestamp_imu_ < lidar_end_time_...");
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer_.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
    {
        imu_time = imu_buffer_.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time_) break;
        meas.imu.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_flg_ = false;
    return true;
}

void FastLIO::publishFrameWorld(const ros::Publisher & cloud_full_pub_)
{
    if(scan_pub_en_)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en_ ? feats_undistort_ : feats_down_body_);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        cloud_full_pub_.publish(laserCloudmsg);
        publish_count_ -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en_)
    {
        int size = feats_undistort_->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort_->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0  && scan_wait_num >= pcd_save_interval_)
        {
            pcd_index_ ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index_) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void FastLIO::publishFrameBody(const ros::Publisher & body_cloud_full_pub_)
{
    int size = feats_undistort_->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort_->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "body";
    body_cloud_full_pub_.publish(laserCloudmsg);
    publish_count_ -= PUBFRAME_PERIOD;
}

void FastLIO::publishEffectWorld(const ros::Publisher & cloud_effect_pub_)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num_, 1));
    for (int i = 0; i < effct_feat_num_; i++)
    {
        RGBpointBodyToWorld(&laser_cloud_ori_->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudFullRes3.header.frame_id = "camera_init";
    cloud_effect_pub_.publish(laserCloudFullRes3);
}

void FastLIO::publishMap(const ros::Publisher & map_cloud_pub_)
{
    int current_kdtree_size = ikdtree_->size();
    // cout<<">>>>>>> current kdtree size: "<< current_kdtree_size << endl;

    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*feats_from_map_, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudMap.header.frame_id = "camera_init";
    map_cloud_pub_.publish(laserCloudMap);
}

template<typename T>
void FastLIO::setPoseStamp(T & out)
{
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = geo_quat_.x;
    out.pose.orientation.y = geo_quat_.y;
    out.pose.orientation.z = geo_quat_.z;
    out.pose.orientation.w = geo_quat_.w;
    
}

void FastLIO::publishOdometry(const ros::Publisher & odom_pub_)
{
    odom_aft_mapped_.header.frame_id = "camera_init";
    odom_aft_mapped_.child_frame_id = "body";
    odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_);// ros::Time().fromSec(lidar_end_time_);
    setPoseStamp(odom_aft_mapped_.pose);
    odom_pub_.publish(odom_aft_mapped_);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped_.pose.covariance[i*6 + 0] = P(k, 3);
        odom_aft_mapped_.pose.covariance[i*6 + 1] = P(k, 4);
        odom_aft_mapped_.pose.covariance[i*6 + 2] = P(k, 5);
        odom_aft_mapped_.pose.covariance[i*6 + 3] = P(k, 0);
        odom_aft_mapped_.pose.covariance[i*6 + 4] = P(k, 1);
        odom_aft_mapped_.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, \
                                    odom_aft_mapped_.pose.pose.position.y, \
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odom_aft_mapped_.header.stamp, "camera_init", "body" ) );
}

void FastLIO::publishPath(const ros::Publisher path_pub_)
{
    setPoseStamp(msg_body_pose_);
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path_.poses.push_back(msg_body_pose_);
        path_pub_.publish(path_);
    }
}

} // end of namespace fast_lio

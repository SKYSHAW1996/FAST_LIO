// #include <so3_math.h>
// #include "preprocess.h"
// #include <ikd-Tree/ikd_Tree.h>

#include "laserMapping.h"
#include <functional>

namespace fast_lio{

std::mutex FastLIO::FastLIO_Mutex;
shared_ptr<FastLIO> FastLIO::FastLIO_instance = nullptr;

std::shared_ptr<FastLIO> FastLIO::getFastLIO(ros::NodeHandle nh) {
    // std::cout << "Let's get a FastLIO instance." << std::endl;
    if (FastLIO_instance == nullptr) {
        std::unique_lock<std::mutex> lock(FastLIO_Mutex);
        if (FastLIO_instance == nullptr) {
            auto temp = std::shared_ptr<FastLIO>(new FastLIO(nh));
            // std::cout << "Let's get a FastLIO instance with param handler." << std::endl;
            FastLIO_instance = temp;
        }
    }
    return FastLIO_instance;
}

std::shared_ptr<FastLIO> FastLIO::getFastLIO() {
    // std::cout << "try to get a FastLIO instance without roshandler." << std::endl;
    if (FastLIO_instance == nullptr) {
        std::unique_lock<std::mutex> lock(FastLIO_Mutex);
        if (FastLIO_instance == nullptr) {
            ros::NodeHandle nh;
            auto temp = std::shared_ptr<FastLIO>(new FastLIO(nh));
            // std::cout << "Let's get a FastLIO instance with tmp handler." << std::endl;
            FastLIO_instance = temp;
        }
    }
    return FastLIO_instance;
}

FastLIO::FastLIO(ros::NodeHandle nh) : nh_(nh), p_pre(std::make_shared<Preprocess>()), p_imu(std::make_shared<ImuProcess>())
{
    // std::vector<float> res_last_tmp(100000, 0.0);
    // std::vector<bool> point_selected_surf_tmp(100000, true);
    res_last.clear();
    point_selected_surf.clear();
    res_last.resize(100000, 0.0);
    point_selected_surf.resize(100000, true);

    feats_down_body.reset(new PointCloudXYZI());
    feats_down_world.reset(new PointCloudXYZI());
    normvec.reset(new PointCloudXYZI(100000, 1));
    laserCloudOri.reset(new PointCloudXYZI(100000, 1));
    corr_normvect.reset(new PointCloudXYZI(100000, 1));

    nh_.param<bool>("publish/path_en",path_en, true);
    nh_.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh_.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh_.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh_.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh_.param<string>("map_file_path",map_file_path,"");
    nh_.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh_.param<string>("common/imu_topic", imu_topic,"/imu/data");
    nh_.param<bool>("common/time_sync_en", time_sync_en, false);
    nh_.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh_.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh_.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh_.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh_.param<double>("cube_side_length",cube_len,200);
    nh_.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh_.param<double>("mapping/fov_degree",fov_deg,180);
    nh_.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh_.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh_.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh_.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh_.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh_.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh_.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh_.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh_.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh_.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh_.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh_.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh_.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh_.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh_.param<int>("pcd_save/interval", pcd_save_interval, -1);

    nh_.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh_.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    cout<<"cube_side_length "<<cube_len<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    featsFromMap.reset(new PointCloudXYZI());
    feats_undistort.reset(new PointCloudXYZI());

    pcl_wait_pub.reset(new PointCloudXYZI());
    pcl_wait_save.reset(new PointCloudXYZI());
    _featsArray.reset(new PointCloudXYZI());

    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    // memset(res_last, -1000.0f, sizeof(res_last));

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    sub_pcl = nh_.subscribe(lid_topic, 200000, &FastLIO::standard_pcl_cbk, this);
    sub_imu = nh_.subscribe(imu_topic, 200000, &FastLIO::imu_cbk, this);

    pubLaserCloudFull = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1);
    pubLaserCloudFull_body = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 1);
    pubLaserCloudEffect = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 1);
    pubLaserCloudMap = nh_.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1);
    pubOdomAftMapped = nh_.advertise<nav_msgs::Odometry>("/Odometry", 1);
    pubPath = nh_.advertise<nav_msgs::Path>("/path", 1);

//------------------------------------------------------------------------------------------------------
    sync_packages_timer_ = nh_.createTimer(::ros::Duration(0.005), &FastLIO::process_data_packages, this);
    signal(SIGINT, SigHandle);
    // ros::Rate rate(1000);
    // bool status = ros::ok();
    // while (status)
    // {
    //     if (flg_exit) break;
    //     ros::spinOnce();
        
    //     // process_data_packages()

    //     status = ros::ok();
    //     rate.sleep();
    // }

}

FastLIO::~FastLIO() {}

void FastLIO::process_data_packages(const ::ros::TimerEvent& timer_event)
{
    if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            /*** initialize the map kdtree ***/
            if(ikdtree->Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree->set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree->Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree->validnum();
            kdtree_size_st = ikdtree->size();
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;
            // cout<<"kdtree_size_st "<<kdtree_size_st<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);

            if(1) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree->PCL_Storage);
                ikdtree->flatten(ikdtree->Root_Node, ikdtree->PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree->PCL_Storage;
            }

            // pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en) {
                if(++keyframe_pulse_cout > 10){
                    publish_frame_world(pubLaserCloudFull);
                    keyframe_pulse_cout = 0;
                }
            }
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            publish_map(pubLaserCloudMap);

        }
}

void FastLIO::save_pcd()
{
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
}

void FastLIO::dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // gravity  
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
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void FastLIO::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void FastLIO::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void FastLIO::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void FastLIO::points_cache_collect()
{
    PointVector points_history;
    ikdtree->acquire_removed_points(points_history);
    _featsArray->clear();
    for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
    // cout<<">>>>>>> kdtree acquire_removed_points size: "<< _featsArray->size() << endl;

}

void FastLIO::lasermap_fov_segment()
{   
    // 清空上一次需要移除的区域的数据量
    cub_needrm.clear(); 
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // not used
    // 获取 world 系下 lidar 位置
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 获取当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（例如1.0*100m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;

    // 计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    // points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) {
        kdtree_delete_counter = ikdtree->Delete_Point_Boxes(cub_needrm);
        // std::cout<<">>>>>>> kdtree_delete_counter: "<< kdtree_delete_counter << std::endl;
    }
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void FastLIO::map_incremental()
{
    PointVector PointToAdd;                 //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample;      //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    //根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            
            // mid_point即为该特征点所属的栅格的中心点坐标
            PointType downsample_result, mid_point; 
            // filter_size_map_min是地图体素降采样的栅格边长
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            // 当前点与box中心的距离
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            // 判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                // 若三个方向距离都大于地图栅格半轴长，无需降采样
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
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
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            //如果周围没有点或者没有初始化EKF，则加入到PointToAdd中
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    int add_point_size_1 = ikdtree->Add_Points(PointToAdd, true);
    int add_point_size_2 = ikdtree->Add_Points(PointNoNeedDownsample, false); 
    add_point_size = add_point_size_1 + add_point_size_2;
    // std::cout << "kdtree_incremental add_point_size: " << add_point_size << std::endl;
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

void FastLIO::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void FastLIO::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(time_diff_lidar_to_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(time_diff_lidar_to_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool FastLIO::sync_packages(MeasureGroup &meas)
{
    // 此处的sync限制了发布的频率，因为当lidar_buffer为空时，sync_packages返回为false，并不会利用新的imu数据进行predict来发布位姿的递推值
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    // ROS_INFO(">>>>>>>>>> lidar_buffer size: %d", lidar_buffer.size());
    // ROS_INFO(">>>>>>>>>> imu_buffer size: %d", imu_buffer.size());

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        // ROS_WARN("last_timestamp_imu < lidar_end_time...");
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void FastLIO::publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
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
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void FastLIO::publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void FastLIO::publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void FastLIO::publish_map(const ros::Publisher & pubLaserCloudMap)
{
    int current_kdtree_size = ikdtree->size();
    // cout<<">>>>>>> current kdtree size: "<< current_kdtree_size << endl;

    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void FastLIO::set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void FastLIO::publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void FastLIO::publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

} // end of namespace fast_lio

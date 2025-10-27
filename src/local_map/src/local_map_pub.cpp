#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include <grid_map_octomap/GridMapOctomapConverter.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

#include "ellipse.hpp"
#include "DBSCAN.hpp"
#include "KM.hpp"

using namespace grid_map;
using namespace std;

GridMap map_({"elevation", "local_lidar", "gradient_map"});

pcl::PointCloud<pcl::PointXYZ> velodyne_cloud;
pcl::PointCloud<pcl::PointXYZ> velodyne_cloud_filter;
pcl::PointCloud<pcl::PointXYZ> velodyne_cloud_global;

pcl::PassThrough<pcl::PointXYZ> pass_x;
pcl::PassThrough<pcl::PointXYZ> pass_y;
pcl::VoxelGrid<pcl::PointXYZ> sor;

tf::TransformListener *listener_ptr = nullptr;
tf::StampedTransform base_link_transform;
tf::StampedTransform lidar_link_transform;
Eigen::Vector2d robot_position2d;
geometry_msgs::PoseStamped current_pose;

// ros sub & pub
ros::Subscriber velodyne_sub;

ros::Publisher for_obs_track_pub;

KMAlgorithm KM;

// DBSCAN param
float DBSCAN_R;
int DBSCAN_N;

float step_height;
float obs_height;

int block_size;
int block_num;

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  current_pose = *msg;
}

void pcd_transform()
{
  geometry_msgs::PoseStamped pose;
  {
    pose = current_pose;
  }

  pass_x.setInputCloud(velodyne_cloud.makeShared());
  pass_x.filter(velodyne_cloud);

  pass_y.setInputCloud(velodyne_cloud.makeShared());
  pass_y.filter(velodyne_cloud);

  velodyne_cloud_filter.clear();
  sor.setInputCloud(velodyne_cloud.makeShared());
  sor.filter(velodyne_cloud_filter);

  Eigen::Affine3d affine_transform = Eigen::Affine3d::Identity();

  // 设置位置
  affine_transform.translation() << 
      pose.pose.position.x,
      pose.pose.position.y,
      pose.pose.position.z;
  
  // 设置旋转
  Eigen::Quaterniond eigen_q(
    pose.pose.orientation.w,
    pose.pose.orientation.x,
    pose.pose.orientation.y,
    pose.pose.orientation.z
  );
  eigen_q.normalize(); // 确保单位四元数

  // 直接设置旋转部分
  affine_transform.linear() = eigen_q.toRotationMatrix();

  pcl::transformPointCloud(velodyne_cloud_filter, velodyne_cloud_global, affine_transform);
}

void lidar2gridmap(Eigen::MatrixXf &lidar_data_matrix)
{
  int col = lidar_data_matrix.cols();
  int row = lidar_data_matrix.rows();
  for (const auto &pt : velodyne_cloud_global)
  {
    int j = (pt.x - robot_position2d.x()) * _inv_resolution + col * 0.5;
    j = min(max(j, 0), row - 1);
    int k = (pt.y - robot_position2d.y()) * _inv_resolution + row * 0.5;
    k = min(max(k, 0), col - 1);

    if (std::isnan(lidar_data_matrix(row - 1 - j, col - 1 - k)))
      lidar_data_matrix(row - 1 - j, col - 1 - k) = pt.z;
    if (lidar_data_matrix(row - 1 - j, col - 1 - k) < pt.z)
      lidar_data_matrix(row - 1 - j, col - 1 - k) = pt.z;
  }
}

Eigen::MatrixXf map_interpolation(const Eigen::MatrixXf &map_data)
{
  int col = map_data.cols(), row = map_data.rows();
  Eigen::MatrixXf map_interpolation(map_data);
  for (int i = 1; i < row - 1; i++)
  {
    for (int j = 1; j < col - 1; j++)
    {
      if (isnan(map_data(i, j)))
      {
        int count = 0;
        float height = 0;
        for (int k = 0; k <= 2; k++)
        {
          for (int q = 0; q <= 2; q++)
          {
            if (!isnan(map_data(i - 1 + k, j - 1 + q)))
            {
              count++;
              height += map_data(i - 1 + k, j - 1 + q);
            }
          }
        }
        map_interpolation(i, j) = (count > 0) ? height / count : NAN;
      }
    }
  }
  return map_interpolation;
}

void map_inflate_block(Eigen::MatrixXf &dst, const Eigen::MatrixXf &src, int startRow, int startCol, int radius)
{
  for (int k = 0; k <= 2 * radius; k++)
  {
    for (int q = 0; q <= 2 * radius; q++)
    {
      if (isnan(src(startRow - radius + k, startCol - radius + q)))
      {
        dst(startRow - radius + k, startCol - radius + q) = src(startRow, startCol);
      }
    }
  }
}

Eigen::MatrixXf map_inflate(const Eigen::MatrixXf &map_data)
{
  int col = map_data.cols(), row = map_data.rows();
  Eigen::MatrixXf map_inflated(map_data);
  for (int i = 3; i < row - 3; i++)
  {
    for (int j = 3; j < col - 3; j++)
    {
      if (isnan(map_data(i, j)))
        continue;

      double dis = sqrt((i - col / 2) * (i - col / 2) + (j - col / 2) * (j - col / 2));
      int radius;
      if (dis < col / 3)
        radius = 1;
      else if (dis < col * 0.45)
        radius = 2;
      else
        radius = 3;
      map_inflate_block(map_inflated, map_data, i, j, radius);
    }
  }
  return map_inflated;
}

Eigen::MatrixXf gradient_map_processing(Eigen::MatrixXf &map_data, vector<DBSCAN::Point> &dataset)
{
  const float threshold = -1.25;
  int col = map_data.cols(), row = map_data.rows();
  Eigen::MatrixXf gradient_map(row, col);
  gradient_map.setOnes();
  gradient_map *= threshold;
  DBSCAN::Point obs;

  for (int i = 1; i < row - 1; i++)
  {
    for (int j = 1; j < col - 1; j++)
    {
      bool has_nan_value = false;
      for (int p = -1; p <= 1; p++)
      {
        for (int q = -1; q <= 1; q++)
        {
          if (isnan(map_data(i + p, j + q)))
          {
            gradient_map(i, j) = threshold;
            has_nan_value = true;
          }
        }
      }
      if (!has_nan_value)
      {
        float sobel_x = map_data(i + 1, j + 1) - map_data(i - 1, j + 1) + map_data(i + 1, j) - map_data(i - 1, j) +
                        map_data(i + 1, j - 1) - map_data(i - 1, j - 1);
        float sobel_y = map_data(i - 1, j + 1) - map_data(i - 1, j - 1) + map_data(i, j + 1) - map_data(i, j - 1) +
                        map_data(i + 1, j + 1) - map_data(i + 1, j - 1);
        gradient_map(i, j) = sqrt(sobel_x * sobel_x + sobel_y * sobel_y) + threshold;
        if (gradient_map(i, j) > obs_height + threshold)
        {
          obs.x = i;
          obs.y = j;
          dataset.push_back(obs);
        }
        else
        {
          Eigen::MatrixXf::Index minRow, minCol;
          Eigen::MatrixXf block = map_data.block(i / block_size, j / block_size, block_size, block_size);
          float min = block.minCoeff(&minRow, &minCol);
          if (map_data(i, j) - min > step_height)
          {
            obs.x = i;
            obs.y = j;
            dataset.push_back(obs);
          }
        }
      }
    }
  }

  return gradient_map;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "local_map_pub");
  ros::NodeHandle nh("~");

  for_obs_track_pub = nh.advertise<std_msgs::Float32MultiArray>("for_obs_track", 1);
  velodyne_sub = nh.subscribe<sensor_msgs::PointCloud2>("/ouster/points", 1, [&](sensor_msgs::PointCloud2::ConstPtr msg)
                                                        { 
  pcl::fromROSMsg(*msg, velodyne_cloud);    
  });
  ros::Subscriber pose_sub = nh.subscribe("/robot/dlio/odom_node/pose", 10, poseCallback);

  // param
  nh.param<float>("localmap_x_size", localmap_x_size, 10);
  nh.param<float>("localmap_y_size", localmap_y_size, 10);
  nh.param<float>("resolution", resolution, 0.1);

  nh.param<float>("obs_height", obs_height, 0.4);
  nh.param<float>("step_height", step_height, 0.5);
  nh.param<float>("DBSCAN_R", DBSCAN_R, 5.0);
  nh.param<int>("DBSCAN_N", DBSCAN_N, 5);

  nh.param<int>("block_size", block_size, localmap_x_size * _inv_resolution * 0.2);
  nh.param<int>("block_num", block_num, 5);

  _inv_resolution = 1 / resolution;
  int map_index_len = localmap_x_size * _inv_resolution;

  tf::TransformListener listener;
  listener_ptr = &listener;

  // initial
  map_.setFrameId("os_sensor");
  map_.setGeometry(Length(localmap_x_size, localmap_y_size), resolution);
  Eigen::MatrixXf lidar_pcd_matrix(map_index_len, map_index_len);
  Eigen::MatrixXf map_interpolate(map_index_len, map_index_len);
  Eigen::MatrixXf gradient_map(map_index_len, map_index_len);

  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(-localmap_x_size / 2, localmap_x_size / 2);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(-localmap_y_size / 2, localmap_y_size / 2);

  sor.setLeafSize(0.05f, 0.05f, 0.05f);

  // main loop
  ros::Rate rate(11);
  while (ros::ok())
  {
    robot_position2d << current_pose.pose.position.x, current_pose.pose.position.y;
    pcd_transform();
    map_.setPosition(robot_position2d);
    map_.clear("local_lidar");
    lidar_pcd_matrix = map_.get("local_lidar");
    lidar2gridmap(lidar_pcd_matrix);
    map_.add("local_lidar", lidar_pcd_matrix);

    map_interpolate = map_interpolation(lidar_pcd_matrix);
    map_interpolate = map_inflate(map_interpolate);
    map_.add("elevation", map_interpolate);

    // obs map
    vector<DBSCAN::Point> non_clustered_obs;
    gradient_map = gradient_map_processing(map_interpolate, non_clustered_obs);

    // DBSCAN
    DBSCAN DS(DBSCAN_R, DBSCAN_N, non_clustered_obs);
    vector<Obstacle> clustered_obs(DS.cluster_num);
    for (const auto &obs : non_clustered_obs)
    {
      if (obs.obsID > 0)
      {
        // gradient_map(x, y) = -0.3;
        gradient_map(obs.x, obs.y) = -0.3f;
        clustered_obs[obs.obsID - 1].emplace_back(obs.x, obs.y);
      }
    }    

    // get MSE
    vector<Ellipse> ellipses_array = get_ellipse_array(clustered_obs, map_);
    KM.tracking(ellipses_array);
    ab_variance_calculation(ellipses_array);

    // publish
    std_msgs::Float32MultiArray for_obs_track;
    for (const auto &ellipse : ellipses_array)
    {
      if (ellipse.label == 0)
        continue;

      for_obs_track.data.push_back(ellipse.cx);
      for_obs_track.data.push_back(ellipse.cy);
      for_obs_track.data.push_back(ellipse.semimajor);
      for_obs_track.data.push_back(ellipse.semiminor);
      for_obs_track.data.push_back(ellipse.theta);
      for_obs_track.data.push_back(ellipse.label);
      for_obs_track.data.push_back(ellipse.variance);
    }
    for_obs_track_pub.publish(for_obs_track);


    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

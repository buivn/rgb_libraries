#ifndef RGB_LIBS_H_
#define RGB_LIBS_H_

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include "rgb_libs/image_save.h"
#include "rgb_libs/boundingBox.h"

using EvXf = Eigen::VectorXf;
using Ev4f = Eigen::Vector4f;
using Ev3f = Eigen::Vector3f;
using EmXf = Eigen::MatrixXf;
using Em4f = Eigen::Matrix4f;
using SvEv3f = std::vector<Eigen::Vector3f>;

namespace rgb_libs
{
  class rgbLibs
  {
    public:     // public function
      // constructor with a point cloud topic
      rgbLibs(std::string topic_name);
      // constructor with a pcd file
      rgbLibs(std::string file_name, int check);
      // ~rgbLibs();
      
      // list of displaying functions


      // initialize function
      void initPublisher();
      void initSubscriber();
      void initServer();

      // list of propressing functions
      Eigen::Vector2i transform_point3D_into2D(Ev4f pc_coord, Eigen::Matrix3f cmatrix);
      void draw_smallBox_onObject(Eigen::MatrixX2i pix_poss, const std::string& filename);
      int select_object(Eigen::Vector2i grasp_pos, Eigen::Matrix2Xi ob_centroids);

      // list of Callbackfunction
      void imageCallback(const sensor_msgs::ImageConstPtr& msg);
      bool check_and_save(rgb_libs::image_save::Request &req, rgb_libs::image_save::Response &res);


    private:
      
      ros::NodeHandle nh_; // general ROS NodeHandle - used for pub, sub, advertise ...
      ros::NodeHandle nh_private_; // private ROS NodeHandle - used to get parameter from server
      // ROS Publisher
      ros::Publisher rgbLibs_pose_pub_;
      // ROS Subscribers
      ros::Subscriber rgbLibs_imageCam_sub_;
      // ROS Service Server
      ros::ServiceServer rgbLibs_getImage_ser_;

      // Topic name
      std::string imageTopic_;
      // Create a container for the input point cloud data.
      cv::Mat image_input_;
      Eigen::Matrix3f cam_matrix_;
  };
}

#endif /* RGB_LIBS_H_ */

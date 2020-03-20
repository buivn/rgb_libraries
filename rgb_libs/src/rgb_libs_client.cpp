#include "ros/ros.h"
#include "rgb_libs/image_save.h"
#include "rgb_libs/boundingBox.h"
#include <string>
#include <cstdlib>
// #include <ros/console.h>






int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgb_libs_client");
  ros::NodeHandle n;
  ros::NodeHandle n_para("~");
  // std::string serviceName = "image_cmd";
  std::string serviceName;
  // n_para.param("serviceName", serviceName, std::string("image_capture_py"));
  n_para.param("serviceName", serviceName, std::string("boundingBox"));
  // ROS_INFO("serviceName = %s", serviceName.c_str());
  
  bool wait = ros::service::waitForService(serviceName, 5000);
  // ros::ServiceClient client2 = n.serviceClient<rgb_libs::image_save>(serviceName);
  // rgb_libs::image_save srv;
  // srv.request.cmd = true;
  // srv.request.num_name = std::to_string(i);
  // srv.request.path = "/home/buivn/bui_ws/src/rgb_libs/images/";

  ros::ServiceClient client2 = n.serviceClient<rgb_libs::boundingBox>(serviceName);
  rgb_libs::boundingBox srv;
  srv.request.cmd = true;
  srv.request.topic = true;

  ros::Rate loop_rate(0.1);
  // int i = 0;
  while(ros::ok())
  {
    // if (!ros::service::exists(serviceName, wait)) {    
    if (client2.call(srv))
      for (int k=0; k< srv.response.vecCen.size()/2; k++) {
        ROS_ERROR("Centroid %d: x = %d", k, srv.response.vecCen[2*k]);
        ROS_ERROR("Centroid %d: y = %d", k, srv.response.vecCen[2*k+1]);
      }
      // if (srv.response.result == 1) {
      //   ROS_INFO("Capture an image: %d", i);
      // } 
      // else if (srv.response.result == 2)
      //   ROS_INFO("server was called, but image was not requested");
      // else ROS_INFO("Not capturing successfully");
    else
    {
      ROS_ERROR("Failed to call service ");
      return 1;
    }
    ros::spinOnce();
    loop_rate.sleep();
  } 
  return 0;
}
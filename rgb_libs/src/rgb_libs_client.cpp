#include "ros/ros.h"
#include "rgb_libs/image_cmd.h"
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
  // n.param<std::string>("serviceName", serviceName, std::string("image_capture_py"));
  n_para.param("serviceName", serviceName, std::string("image_capture_py"));
  // ROS_INFO("serviceName = %s", serviceName.c_str());
  // ROS_ERROR(serviceName.c_str());
  
  bool wait = ros::service::waitForService(serviceName, 5000);
  // ros::ServiceClient client1 = n.serviceClient<rgb_libs::image_cmd>("image_cmd");
  ros::ServiceClient client2 = n.serviceClient<rgb_libs::image_cmd>(serviceName);
  rgb_libs::image_cmd srv;
  ros::Rate loop_rate(0.2);
  srv.request.cmd = true;
  srv.request.path = "/home/buivn/bui_ws/src/rgb_libs/images/";
  
  // srv.request.topic = false;


  int i = 0;
  while(ros::ok())
  {
    // if (!ros::service::exists(serviceName, wait)) {
    if (i<6) {
      srv.request.num_name = std::to_string(i);
      if (client2.call(srv))
        if (srv.response.result == 1) {
          ROS_INFO("Capture an image: %d", i);
          // client.call(srv); 
        } 
        else if (srv.response.result == 2)
          ROS_INFO("server was called, but image was not requested");
        else ROS_INFO("Not capturing successfully");
      else
      {
        ROS_ERROR("Failed to call service ");
        return 1;
      }
    }
    i += 1;

    ros::spinOnce();
    loop_rate.sleep();
  } 


  return 0;
}
#include <sstream>
#include <rgb_libs/rgb_libs.h>

namespace rgb_libs
{

// constructor
rgbLibs::rgbLibs(std::string topic_name) : 
  nh_(ros::NodeHandle()),
  nh_private_(ros::NodeHandle("~"))
{
  imageTopic_ = topic_name;
  // new calibration file 05-02-2019 from Asus Xtion Pro Live
  cam_matrix_ << 550.1966, 0.0, 310.8416, 0.0, 551.3066, 243.986948, 0.0, 0.0, 1.0;
  // initialize other element
  // initPublisher();
  initSubscriber();
  initServer();
}

// the second constructor
rgbLibs::rgbLibs(std::string file_name, int check):
  nh_(ros::NodeHandle()),
  nh_private_(ros::NodeHandle("~"))
{
  // new calibration file 05-02-2019 from Asus Xtion Pro Live
  cam_matrix_ << 550.1966, 0.0, 310.8416, 0.0, 551.3066, 243.986948, 0.0, 0.0, 1.0;

  // initialize other element
  // initPublisher();
  initSubscriber();
  initServer();
}

/********************************************************************************
** Init Functions
********************************************************************************/
// void PahaP::initPublisher()
// {
  // ros message publisher
  // pahap_pose_pub_ = nh_.advertise<open_manipulator_msgs::OpenManipulatorState>("states", 10);
// }
void rgbLibs::initSubscriber()
{
  // ros message subscriber
  rgbLibs_imageCam_sub_ = nh_.subscribe (imageTopic_, 1, &rgbLibs::imageCallback, this);
}

void rgbLibs::initServer()
{
  rgbLibs_getImage_ser_ = nh_.advertiseService("image_cmd", &rgbLibs::check_and_save, this);
}

//callback to get camera data through "image_pub" topic
void rgbLibs::imageCallback(const sensor_msgs::ImageConstPtr& msg){   
  try
  {
      // ROS_INFO("Checking checking chekcing");
      image_input_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

bool rgbLibs::check_and_save(rgb_libs::image_cmd::Request &req, rgb_libs::image_cmd::Response &res){
  if (req.cmd) {
    //image name composed by path (finished with "/")+ capture angle+extension
    std::string im_name = req.path + req.num_name+ ".png";
    //checking if the picture has a valid content,
    //otherwise system would failed and stop trying to write the image
    if(!image_input_.empty()) {
      if (!cv::imwrite (im_name, image_input_)) {
        res.result = 0;
        std::cout<<"Image can not be saved as '"<<im_name<<"'\n";
      }
      else {
        // represent success to save the image
        std::cout<<"Image saved in '"<<im_name<<"'\n";
        res.result = 1;
      }
    }
    else {
      // represent fail to save the image
      res.result = 0;
      ROS_ERROR("Failed to save image\n");
    }
  }
  else {
    // represent that server was called, but image was not requested
    res.result = 2;
  }
  return true;
}

Eigen::Vector2i rgbLibs::transform_point3D_into2D(Ev4f pc_coord, Eigen::Matrix3f cmatrix) {
    Eigen::Vector2i pixel_pos;
    //std::cout << "inside the point3Dto_pixel" << std::endl;
    pixel_pos(0) = (int)(pc_coord(0)*cmatrix(0,0)/pc_coord(2) + cmatrix(0,2));
    pixel_pos(1) = (int)(pc_coord(1)*cmatrix(1,1)/pc_coord(2) + cmatrix(1,2));
    //std::cout << "inside the point3Dto_pixel" << std::endl;
    return pixel_pos;
}


void rgbLibs::draw_smallBox_onObject(Eigen::MatrixX2i pix_poss, const std::string& filename) {
  cv::Mat image;
  image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  cv::Point pt1, pt2;
  int size = 10;
  int clusternumber = pix_poss.size()/2;
  //std::cout << "the number of point centroid: " << clusternumber << std::endl;
  for (int it = 0; it < clusternumber; ++it) {
      pt1.x = pix_poss(it,0) - size;
      pt1.y = pix_poss(it,1) - size;
      pt2.x = pix_poss(it,0) + size;
      pt2.y = pix_poss(it,1) + size;
      cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 255 ), -1, 8);
  }
  cv::namedWindow("Display window");
  cv::imshow("Display window", image);
  cv::waitKey(15000);
}

int rgbLibs::select_object(Eigen::Vector2i grasp_pos, Eigen::Matrix2Xi ob_centroids)
{
  int distance, distance_x, distance_y, threshold, ob_grasp;
  threshold = 40;
  ob_grasp = 20;
  int clusternumber = ob_centroids.size()/2;
  for (int it = 0; it < clusternumber; ++it) {
    distance_x = abs(grasp_pos(0) - ob_centroids(it,0));
    distance_y = abs(grasp_pos(1) - ob_centroids(it,1));
    distance = pow(distance_x, 2) + pow(distance_y, 2);
    //std::cout <<  "The distance is: " << distance << std::endl;
    if (distance < pow(threshold,2)) {
      ob_grasp = it;
      std::cout <<  "The object to be grasped is: " << it+1 << std::endl;
      break;
    }
  }
  return ob_grasp;
}


}     // end of namespace

int main(int argc, char **argv)
{
  // pointcloudServer pclServer;
  ros::init(argc, argv, "rgb_libs_processing");
  ros::NodeHandle nh_main("~");
  
  std::string rgb_topicName, rgb_fileAddress;
  
  nh_main.param("rgb_topicName", rgb_topicName, std::string("/camera_remote/rgb/image_rect_color"));
  nh_main.param("rgb_fileAddress", rgb_fileAddress, std::string("/home/buivn/bui_ws/src/rgbLibs/scripts/testImg/test3.jpg"));

  // select the object to grasp
  // std::vector<int> haipixel;
  // node.getParam("selected_2Dpixel", haipixel);  
  // selected_2Dpixel << haipixel[0], haipixel[1];
  // int  distance1;
  // nh_main.param("distance1", distance1, 51);


  ros::Rate loop_rate(100);
  // std::string imageTopic = "/camera_remote/rgb/image_rect_color";


  rgb_libs::rgbLibs image_process(rgb_topicName);
  // rgb_libs::rgbLibs pahap_process(filename, 1);

  ros::spin();
  // while (ros::ok())
  // {
  //   ros::spinOnce();
  //   loop_rate.sleep();
  // }

  return 0;
}

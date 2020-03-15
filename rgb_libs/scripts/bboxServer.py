#!/usr/bin/env python

import rospy
import roslib
import sys
from std_msgs.msg import String
import numpy as np
from rgb_libs.srv import image_cmd
from sensor_msgs.msg import Image
from numpy import expand_dims
import cv2
from cv_bridge import CvBridge, CvBridgeError

class boundbox_server:
  # cv::Mat image_
  bridge = CvBridge()
  def __init__(self):
    self.image_topic = ""
    self.image_file = ""
    self.bridge = CvBridge()
    self.bb_image = np.zeros((480,640,3), np.uint8)
    


  def parseUserArgument(self):
    for i in range(0,len(sys.argv)):
      if sys.argv[i] == '-topic':
        if len(sys.argv) > i+1:
          # if self.param_name != "" or self.database_name != "":
          #   rospy.logerr("Error: you cannot specify parameter if file or database name is given, must pick one source of model xml")
          #   sys.exit(0)
          # else:
          self.image_topic = sys.argv[i+1]

      if sys.argv[i] == '-file':
        # if len(sys.argv) > i+1:
        self.image_file = sys.argv[i+1]

  def callSpawnServiceTopic(self):
    rospy.init_node('image_BBox')    # initialize a node
    rospy.Subscriber(self.image_topic, Image, self.imageCallback)   # subscribe a topic
    rospy.Service('image_capture_py', image_cmd, self.serverResponse)   # advertise a service
    r= rospy.Rate(100)
    while not rospy.is_shutdown():
      r.sleep()

  
  def imageCallback(self, data):
    try:
      self.bb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print (e)


  def serverResponse(self, data):
    if data.cmd:
      im_name = data.path + data.num_name+ ".png";
      if not (self.bb_image.size == 0):
        if (not cv2.imwrite (im_name, self.bb_image)):
          return 0    # return the value for the service.response
          # print('Image can not be saved as ', str(im_name))
          rospy.logwarn("Image can not be saved")
        else:
          rospy.loginfo("Image saved in: %s", data.path)
          return 1 # return the value for the service.response
      else: 
        # represent fail to save the image
        return 0  # return the value for the service.response
        logerr("Error: Failed to save image on service: image_capture_py")
    else:
      return 2  # return the value for the service.response

def main(args):
  
  # rgb_topicName = rospy.get_param('rgb_topicName')
  # ROS_INFO(rgb_topicName)
  # rgb_fileAddress = rospy.get_param('rgb_fileAddress')
  # bbServer = boundbox_server("/camera_remote/rgb/image_rect_color", "/home/buivn/bui_ws/src/rgbLibs/scripts/testImg/test3.jpg")
  bbServer = boundbox_server()
  bbServer.parseUserArgument()
  bbServer.callSpawnServiceTopic()
  
  # rospy.init_node('image_BBox', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()



if __name__ == '__main__':

    main(sys.argv)
    # # topic name
    # pub_dzungYolov3Keras = rospy.Publisher('Object_Centroid', String, queue_size=2)
    # # node name
    # rospy.init_node('dzungyolov3keras', anonymous = True)
    # rate = rospy.Rate(0.2)

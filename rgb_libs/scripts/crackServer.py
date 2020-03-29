#!/usr/bin/env python
import rospy
import roslib
import sys
import numpy as np
from sensor_msgs.msg import Image as SImage
from numpy import expand_dims
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
from cv_bridge import CvBridge, CvBridgeError

from crackDetection import *


class crack_server:
  def __init__(self):
    self.image_topic = ""
    self.image_file = ""
    self.basepath = ""
    # self.modelAdress = ""
    # self.modelAdress = ""
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
      if sys.argv[i] == '-basepath':
        # if len(sys.argv) > i+1:
        self.basepath = sys.argv[i+1]
      if sys.argv[i] == '-model':
        self.crackProcess = crack_detection(self.basepath, sys.argv[i+1])

  def callSpawnServiceTopic(self):
    rospy.init_node('crack_detection')    # initialize a node
    # rospy.Subscriber(self.image_topic, SImage, self.imageCallback)   # subscribe a topic
    # rospy.Service('boundingBox', boundingBox, self.serverResponse)   # advertise a service
    r= rospy.Rate(100)
    while not rospy.is_shutdown():
      r.sleep()
 
  def imageCallback(self, data):
    try:
      self.bb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print (e)

  # def serverResponse(self, data):
  #   if data.cmd:


  #   return boundingBoxResponse(cenArr)   # return the value for the service.response

def main(args):
  
  # rgb_topicName = rospy.get_param('rgb_topicName')
  # ROS_INFO(rgb_topicName)
  # rgb_fileAddress = rospy.get_param('rgb_fileAddress')
  # bbServer = boundbox_server("/camera_remote/rgb/image_rect_color", "/home/buivn/bui_ws/src/rgbLibs/scripts/testImg/test3.jpg")
  crackServer = crack_server()
  crackServer.parseUserArgument()
  crackServer.callSpawnServiceTopic()
  crackServer.crackProcess.detect_crack()

  
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

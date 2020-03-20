#!/usr/bin/env python
# load a YOLOv3 Keras model and process an pre-processed image 
# based on https://github.com/experiencor/keras-yolov

import rospy
from std_msgs.msg import String
import numpy as np
from numpy import expand_dims

# packages to process the sensor/camera
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Instantiate CvBridge
bridge = CvBridge()

lastTime = 0.0

def image_callback(msg):
    print("Callback function run!")
    try:
        # Convert ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        check_message = "The objects' Centroids: %s" % centroid_array
        rospy.loginfo(check_message)
        pubImgFrequency.publish(imageData)




if __name__ == '__main__':

    # node name
    rospy.init_node('pubImgFrequency', anonymous = True)
    rate = rospy.Rate(0.1)
    # topic name
    pubImgFrequency = rospy.Publisher('OneImageinTenSecond', String, queue_size=2)
    # Define your image topic
    image_topic = "/camera_remote/rgb/image_rect_color"

    check_saveImg = False
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)    


    rospy.spin()
    
    rate.sleep()
#     #rospy.spin()
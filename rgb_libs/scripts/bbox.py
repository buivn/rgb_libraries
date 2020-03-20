#!/usr/bin/env python
# load a YOLOv3 Keras model and process an pre-processed image 
# based on https://github.com/experiencor/keras-yolov

import rospy
from std_msgs.msg import String
import numpy as np
from numpy import expand_dims
# packages to process the Keras/Yolov3
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
# packages to process the sensor/camera
from sensor_msgs.msg import Image as SImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PImage

# Instantiate CvBridge
bridge = CvBridge()


class BoundBox:
  def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax
    self.objness = objness
    self.classes = classes
    self.label = -1
    self.score = -1

  def get_label(self):
    if (self.label == -1):
        self.label = np.argmax(self.classes)
    return self.label

  def get_score(self):
    if (self.score == -1):
        self.score = self.classes[self.get_label()]
    return self.score


class Box_Processing:
  def __init__(self, model_address):
    self.check_message = ""
    self.process_image = ""
    self.class_threshold = 0.6
    self.anchors = [[116,90, 156,198, 373,326], [30,61,  62,45, 59,119], [10,13, 16,30, 33,23]]
    # self.cenArr = 
    self.labels = ["lotion", "deodorant", "cup", "can"]
    self.input_w = 416
    self.input_h = 416
    self.model = load_model(model_address)

  def _signoid(self, x):
    return 1./(1. + np.exp(-x))

  def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
      grid_h, grid_w = netout.shape[:2]
      nb_box = 3
      netout = netout.reshape((grid_h, grid_w, nb_box, -1))
      nb_class = netout.shape[-1] - 5
      boxes = []
      netout[..., :2] = self._signoid(netout[..., :2])
      netout[..., 4:] = self._signoid(netout[..., 4:])
      netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
      netout[..., 5:] *= netout[..., 5:] > obj_thresh

      for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
          # 4th element is objectness score
          objectness = netout[int(row)][int(col)][b][4]
          #objectness = netout[..., :4]
          
          if(objectness.all() <= obj_thresh): continue
          
          # first 4 elements are x, y, w, and h
          x, y, w, h = netout[int(row)][int(col)][b][:4]

          x = (col + x) / grid_w # center position, unit: image width
          y = (row + y) / grid_h # center position, unit: image height
          w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
          h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
          
          # last elements are class probabilities
          classes = netout[int(row)][col][b][5:]
          
          box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
          #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

          boxes.append(box)

      return boxes

  def correct_yolo_boxes(self, boxes, image_h, image_w, net_h, net_w):
      new_w, new_h = net_w, net_h
      for i in range(len(boxes)):
          x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
          y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
          boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
          boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
          boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
          boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


  def _interval_overlap(self, interval_a, interval_b):
      x1, x2 = interval_a
      x3, x4 = interval_b
      if x3 < x1:
          if x4 < x1:
              return 0
          else:
              return min(x2,x4) - x1
      else:
          if x2 < x3:
               return 0
          else:
              return min(x2,x4) - x3

  def bbox_iou(self, box1, box2):
      intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
      intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
      intersect = intersect_w * intersect_h
      w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
      w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
      union = w1*h1 + w2*h2 - intersect
      return float(intersect) / union
   
  def do_nms(self, boxes, nms_thresh):
      if len(boxes) > 0:
          nb_class = len(boxes[0].classes)
      else:
          return
      for c in range(nb_class):
          sorted_indices = np.argsort([-box.classes[c] for box in boxes])
          for i in range(len(sorted_indices)):
              index_i = sorted_indices[i]
              if boxes[index_i].classes[c] == 0: continue
              for j in range(i+1, len(sorted_indices)):
                  index_j = sorted_indices[j]
                  if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                      boxes[index_j].classes[c] = 0

  #load and prepare an image
  def load_image_fromFile(self, filename, shape):   
      # load the image to get its shape
      image = load_img(filename)  # return a PIL image
      width, height = image.size
      # load the image with the required size (a function of keras)
      image = load_img(filename, target_size =shape)
      # convert the loaded PIL image object into a NumPy array
      image = img_to_array(image) # keras function
      # scale pixel values to [0, 1]
      image = image.astype('float32')
      image /= 255.0
      # add a dimension so that we have one sample ????????? why do we need this dimension
      image = expand_dims(image, 0)
      return image, width, height

  def load_image_fromService(self, image1, shape):   
    # convert an openCV2 to PIL file
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # convert the color.
    im_pil = PImage.fromarray(img)
    width, height = im_pil.size

    # change size of the images
    image2 = cv2.resize(image1, (self.input_h, self.input_w))
    # convert an openCV2 to PIL file
    img = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # convert the color.
    im_pil = PImage.fromarray(img)

    # convert the loaded PIL image object into a NumPy array
    image = img_to_array(im_pil) # keras function
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample ????????? why do we need this dimension
    image = expand_dims(image, 0)
    return image, width, height

  #get all the results above a threshold
  def get_boxes(self, boxes, labels, thresh):
      v_boxes, v_labels, v_scores = list(), list(), list()
      # enumerate all boxes
      for box in boxes:
          # enumerate all possible labels
          for i in range(len(labels)):
              # check if the threshold for this label is high enough
              if box.classes[i] > thresh:
                  v_boxes.append(box)
                  v_labels.append(labels[i])
                  v_scores.append(box.classes[i]*100)
                  # don't break, many labels may trigger for one box
      return v_boxes, v_labels, v_scores

  # draw all results
  def draw_boxes_fromFile(self, filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)

    # get the context for drawing boxes
    ax = pyplot.gca()

    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # calculate the centroid of the box
        xCentroid = int((x1 + x2)/2)
        yCentroid = int((y1 + y2)/2)
        # create the box of the centroid
        rect1 = Rectangle((xCentroid-3, yCentroid-3), 6, 6, fill=True, color='red')
        
        # draw the box
        ax.add_patch(rect)
        ax.add_patch(rect1)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()

    # draw all results
  def draw_boxes_fromService(self, image1, v_boxes, v_labels, v_scores):
    # load the image
    # data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(image1)

    # get the context for drawing boxes
    ax = pyplot.gca()

    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # calculate the centroid of the box
        xCentroid = int((x1 + x2)/2)
        yCentroid = int((y1 + y2)/2)
        # create the box of the centroid
        rect1 = Rectangle((xCentroid-3, yCentroid-3), 6, 6, fill=True, color='red')
        
        # draw the box
        ax.add_patch(rect)
        ax.add_patch(rect1)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()  

  # determine and draw centroid
  def deter_centroid(self, v_boxes1):
    centroid = np.empty(shape=(len(v_boxes1),2), dtype=int)
    # plot each box
    for i in range(len(v_boxes1)):
      box = v_boxes1[i]
      # get centroid coordinates
      centroid[i][0] = int((box.xmin + box.xmax)/2)
      centroid[i][1] = int((box.ymin + box.ymax)/2)
      # centroid.append((xCentroid, yCentroid))
    return centroid

def image_callback(msg):
    print("Received an image!")
    try:
        # Convert ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpg 
        cv2.imwrite('/home/buivn/bui_ws/src/bui_yolov3keras/scripts/testImg/camera_image.jpg', cv2_img)




if __name__ == '__main__':

    # load a Yolov3 model with weights
    # model = load_model('/home/buivn/bui_ws/src/rgb_libs/scripts/models/cv_model_07112019.h5')
    # define the expected input (image) shape (size) for the model
    # input_w, input_h = 416, 416
    # node name
    rospy.init_node('cv_yolov3keras', anonymous = True)
    rate = rospy.Rate(0.1)
    # topic name
    pub_dzungYolov3Keras = rospy.Publisher('Object_Centroid', String, queue_size=2)
    # Define your image topic
    image_topic = "/camera_remote/rgb/image_rect_color"

    check_saveImg = False
    # Set up your subscriber and define its callback
    # rospy.Subscriber(image_topic, SImage, image_callback)

    bPro = Box_Processing('/home/buivn/bui_ws/src/rgb_libs/cnn_models/cv_model_07112019.h5')   


    # rospy.spin()

    while not rospy.is_shutdown():
        # define our new photo
        address = '/home/buivn/bui_ws/src/rgb_libs/images/'
        # photo_filename = address + 'test1.jpg'
        photo_filename = address + 'camera_image.jpg'
        # load and prepare image --- image_w, image_h is the original size of image
        image, image_w, image_h = bPro.load_image_fromFile(photo_filename, (bPro.input_w, bPro.input_h))

        # make prediction (how many objects can be detected in the image)
        yhat = bPro.model.predict(image)
        # summarize the shape of the list of arrays
        #print([a.shape for a in yhat])


        # define the anchors
        # anchors = [[116,90, 156,198, 373,326], [30,61,  62,45, 59,119], [10,13, 16,30, 33,23]]
        # define the probability threshold for detected objects
        # class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += bPro.decode_netout(yhat[i][0], bPro.anchors[i], bPro.class_threshold, bPro.input_h, bPro.input_w)

        # correct the sizes of the bounding boxes for the shape of the image
        bPro.correct_yolo_boxes(boxes, image_h, image_w, bPro.input_h, bPro.input_w)

        # handle the overlapping boxes (for just one object) - suppress non-maximal boxes
        bPro.do_nms(boxes, 0.5)
         
        # define the labels
        # labels = ["lotion", "deodorant", "cup", "can"]

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = bPro.get_boxes(boxes, bPro.labels, bPro.class_threshold)

        # summarize what we found
        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        # centroid_array = bPro.deter_centroid(v_boxes)
        bPro.cenArr = bPro.deter_centroid(v_boxes)
        
        print(bPro.cenArr)

        # draw what we found - the image, the boxes
        bPro.draw_boxes_fromFile(photo_filename, v_boxes, v_labels, v_scores)

    
        bPro.check_message = "The objects' Centroids: %s" % bPro.cenArr
        rospy.loginfo(bPro.check_message)
        pub_dzungYolov3Keras.publish(bPro.check_message)
        rate.sleep()
        rospy.spin()
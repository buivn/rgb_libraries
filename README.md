# rgb libraries
This repos contains all the functions which are used in the projects related to rgb images. It contains several libraries, functions as following\
(1) rgb_libs.cpp: cpp program handles with the camera and service. It saves a rgb file each time the service is called. \
(2) rgb_libs_client.cpp: will call a proxy to the designated service, both roscpp and rospy. \
(3) imageServer.py: similar to rgb_libs.cpp, but wrote in python \
(4) bbox.py: deploy tensorflow/keras to read the model from yolo, and detect the centroid of a object. It is used in the program bboxServer.py \
(5) combine the service with the bounding box program. Each time a service is called, it will detect object and return its centroid.\


import rospy
from point_cloud_classifier.msg import PointsList
from .messages_creator import create_labeled_points_message, create_labels_message, create_point_cloud_message
import ros_numpy
import numpy as np
from time import time


class OutputPointCloudClassifierAdapter(object):
    def __init__(self,
                 labeled_points_topic="/labeled_points"):
        self.__labels_publisher = rospy.Publisher(labeled_points_topic, PointsList, queue_size=10)

    def publish_segmented_point_cloud(self,
                                      points,  # [:, 3]
                                      labels,  # [:, 1]
                                      tf_data,
                                      header
                                      ):
        point_cloud = create_point_cloud_message(points, labels, header)
        labeled_points_message = create_labeled_points_message(point_cloud.header, point_cloud, labels, tf_data)

        self.__labels_publisher.publish(labeled_points_message)

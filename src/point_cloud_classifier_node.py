import rospy
from time import time
import os
from PIL import Image
import numpy as np
from .transformer import transform_pcl


def save_image(image, index, save_path, name):
    image_name = os.path.join(save_path, name + str(index) + ".png")

    rgb_image = Image.fromarray(image)
    rgb_image.save(image_name)


class PointCloudClassifierNode(object):
    def __init__(self,
                 input_point_cloud_classifier_adapter_one,
                 point_cloud_classifier_tester,
                 output_point_cloud_classifier_adapter,
                 test_settings,
                 period=0.1):
        self.input_point_cloud_classifier_adapter_one = input_point_cloud_classifier_adapter_one
        self.point_cloud_classifier_tester = point_cloud_classifier_tester
        self.output_point_cloud_classifier_adapter = output_point_cloud_classifier_adapter

        # Save input images
        self.save_input = test_settings["save_input"]
        self.input_images_saver_path = test_settings["input_save_dir"]

        print("Starting classification")
        self.__timer = rospy.Timer(rospy.Duration(period), self.__timer_callback)

    def __timer_callback(self, _):
        start_time = time()
        # Labeling of point cloud
        image = self.input_point_cloud_classifier_adapter_one.get_image_array()
        points, point_cloud, non_nan_indices = self.input_point_cloud_classifier_adapter_one.get_point_cloud()
        intrinsics = self.input_point_cloud_classifier_adapter_one.get_camera_intrinsics()

        index = self.input_point_cloud_classifier_adapter_one.index
        tf_camera_to_map = self.input_point_cloud_classifier_adapter_one.tf_camera_to_map
        tf_camera_to_base = self.input_point_cloud_classifier_adapter_one.tf_camera_to_base

        if image is None or points is None or intrinsics is None or tf_camera_to_map is None or tf_camera_to_base is None:
            return None

        if self.save_input:
            save_image(image, index, self.input_images_saver_path, "collision_avoid_ds_mars")

        all_labels = -np.ones(480 * 640)
        labels = self.point_cloud_classifier_tester.launch_point_cloud_classifier_tester(image,
                                                                                         points,
                                                                                         intrinsics,
                                                                                         index,
                                                                                         True,
                                                                                         tf_camera_to_base)
        all_labels[non_nan_indices] = labels

        # Filtering far points
        non_far_mask = (points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2) ** 0.5 <= 5
        points = points[non_far_mask]
        labels = labels[non_far_mask]

        points = transform_pcl(tf_camera_to_map, points)
        self.output_point_cloud_classifier_adapter.publish_segmented_point_cloud(points,
                                                                                 labels.reshape(-1, 1),
                                                                                 tf_camera_to_map,
                                                                                 point_cloud.header
                                                                                 )
        # print('Point cloud classifier rate:', time() - start_time)

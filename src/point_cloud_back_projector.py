import numpy as np
from PIL import Image
import os
from .transformer import transform_point, transform_plane, transform_pcl
import rospy


def project_point_cloud(points, intrinsics):
    points = np.array(points)
    if points.ndim == 2:
        x = points[:, 0] / points[:, 2]
        y = points[:, 1] / points[:, 2]
        if intrinsics.model == "RS2_DISTORTION_MODIFIED_BROWN_CONRADY":
            r2 = x ** 2 + y ** 2
            f = 1 + intrinsics.coeffs[0] * r2 + intrinsics.coeffs[1] * r2 ** 2 + intrinsics.coeffs[4] * r2 ** 3
            x *= f
            y *= f
            dx = x + 2 * intrinsics.coeffs[2] * x * y + intrinsics.coeffs[3] * (r2 + 2 * x ** 2)
            dy = y + 2 * intrinsics.coeffs[3] * x * y + intrinsics.coeffs[2] * (r2 + 2 * y ** 2)
            x = dx
            y = dy
        elif intrinsics.model == "RS2_DISTORTION_FTHETA":
            r = np.sqrt(x ** 2 + y ** 2)
            rd = (1.0 / intrinsics.coeffs[0] * np.arctan(2 * r * np.tan(intrinsics.coeffs[0] / 2.0)))
            x *= rd / r
            y *= rd / r

        pixel_x = x * intrinsics.fx + intrinsics.ppx
        pixel_y = y * intrinsics.fy + intrinsics.ppy

        return pixel_x.astype(int), pixel_y.astype(int)


def save_as_image(pixel_point_cloud, image_size, name, saver_path):
    image = np.zeros(image_size, dtype=np.uint8)
    pixel_x = pixel_point_cloud[0]
    pixel_y = pixel_point_cloud[1]
    pixel_x *= int(image_size[0] / 640)
    pixel_y *= int(image_size[1] / 480)
    image[pixel_x, pixel_y] = [255, 255, 255]
    image = Image.fromarray(image)
    image.save(os.path.join(saver_path, name + '.png'))


def filter_falses(labels, points, environment='flat', stone_label=None, diagonal=None):
    if environment == 'flat':
        # Filtering false positives
        labels[np.where(points[:, 2] <= 0.025)] = 0

    elif environment == 'unstructured':
        stone_points = points[labels == stone_label]
        # decimated_points = stone_points[:, 0:2] // 0.06 * 0.06
        # stone_points = np.unique(decimated_points, axis=0)
        mean_coordinates = np.mean(stone_points, axis=0)
        stone_mask = ((points[:, 0] > mean_coordinates[0] - diagonal / 2) *
                      (points[:, 0] < mean_coordinates[0] + diagonal / 2)) * \
                     ((points[:, 1] > mean_coordinates[1] - diagonal / 2) *
                      (points[:, 1] < mean_coordinates[1] + diagonal / 2))
        labeling_mask = ((points[:, 0] > mean_coordinates[0] - diagonal / 1) *
                         (points[:, 0] < mean_coordinates[0] + diagonal / 1)) * \
                        ((points[:, 1] > mean_coordinates[1] - diagonal / 1) *
                         (points[:, 1] < mean_coordinates[1] + diagonal / 1))
        labels_in_mask = labels[stone_mask]
        points_in_mask = points[stone_mask]
        stone_environment_coordinates = points_in_mask[labels_in_mask == 0]
        local_height = np.mean(stone_environment_coordinates[:, 2]) + 0.04
        below_mask = points[:, 2] < local_height
        labels[labeling_mask * below_mask] = 0

    # Updating diagonal
    stone_points = points[labels == stone_label]
    if len(stone_points > 0):
        label_x_dimension = np.max(stone_points[:, 0]) - np.min(stone_points[:, 0])
        label_y_dimension = np.max(stone_points[:, 1]) - np.min(stone_points[:, 1])

        diagonal = (label_x_dimension ** 2 + label_y_dimension ** 2) ** 0.5

    # labels[stone_mask] = 255

    return labels, diagonal


class PointCloudBackProjector(object):
    def __init__(self):
        self.horizontal_dimension = 0.25  # 0.29
        self.vertical_dimension = 0.29  # 0.25
        self.point_cloud_images_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/point_cloud'

        environment_type = rospy.search_param("environment")
        self.environment = rospy.get_param(environment_type)

        if isinstance(self.environment, dict):
            self.environment = "flat"  # default value

    def deproject_point_cloud(self, masks, points, intrinsics, index, tf_camera_to_base, save_point_cloud_as_image):
        labels = np.zeros(points.shape[0], dtype=np.int32)

        pixel_x, pixel_y = project_point_cloud(points, intrinsics)

        # threshold_vector = np.array([self.dimension_x, self.dimension_y, self.dimension_z]).reshape(-1, 1)
        # threshold_vector_c = transform_point(tf_base_to_camera, threshold_vector)
        base_points = transform_pcl(tf_camera_to_base, points)

        # Point cloud deprojection
        for i, mask in enumerate(masks):
            initial_labels = mask[pixel_x, pixel_y]

            stone_points = base_points[np.where(initial_labels == 255)]

            if len(stone_points > 0):
                label_x_dimension = np.max(stone_points[:, 0]) - np.min(stone_points[:, 0])
                label_y_dimension = np.max(stone_points[:, 1]) - np.min(stone_points[:, 1])
                label_z_dimension = np.max(stone_points[:, 2]) - np.min(stone_points[:, 2])

                diagonal = (label_x_dimension ** 2 + label_y_dimension ** 2) ** 0.5

                # Low level plane filtering
                initial_labels, diagonal = filter_falses(initial_labels, base_points, self.environment, 255, diagonal)

                if diagonal < self.horizontal_dimension and label_z_dimension < self.vertical_dimension:
                    labels[np.where(initial_labels == 255)] = 2000 + i
                else:
                    labels[np.where(initial_labels == 255)] = 1000 + i

        if save_point_cloud_as_image:
            save_as_image((pixel_x, pixel_y),
                          (1280, 960, 3),
                          'point_cloud_image' + str(index),
                          self.point_cloud_images_path)

        return labels

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
import tf
import ros_numpy
import pyrealsense2 as rs
import message_filters


class InputPointCloudClassifierAdapter(object):
    def __init__(self,
                 image_topic='/d435_1/color/image_raw',
                 point_cloud_topic='/d435_1/aligned_points',
                 camera_info_topic='/d435_1/color/camera_info',
                 camera_base_tf_topic='/cam_base_1',  # cam_base_1
                 camera_map_tf_topic='/cam_map_1',  # cam_map_1
                 current_frame='d435_1_color_optical_frame',
                 synchronize=True):
        if synchronize:
            image_subscriber = message_filters.Subscriber(image_topic, Image)
            point_cloud_subscriber = message_filters.Subscriber(point_cloud_topic, PointCloud2)
            camera_info_subscriber = message_filters.Subscriber(camera_info_topic, CameraInfo)
            camera_base_tf_subscriber = message_filters.Subscriber(camera_base_tf_topic, Odometry)
            camera_map_tf_subscriber = message_filters.Subscriber(camera_map_tf_topic, Odometry)
            time_synchronizer = message_filters.TimeSynchronizer([image_subscriber,
                                                                  point_cloud_subscriber,
                                                                  camera_info_subscriber,
                                                                  camera_base_tf_subscriber,
                                                                  camera_map_tf_subscriber], 100)

            time_synchronizer.registerCallback(self.__synchronized_callback)
        else:
            image_subscriber = rospy.Subscriber(image_topic, Image, self.__image_callback)
            point_cloud_subscriber = rospy.Subscriber(point_cloud_topic, PointCloud2, self.__point_cloud_callback)
            camera_info_subscriber = rospy.Subscriber(camera_info_topic, CameraInfo, self.__camera_info_callback)

        self.listener = tf.TransformListener()
        self.current_frame = current_frame

        self.image_data = None
        self.point_cloud_data = None
        self.camera_info_data = None
        self.tf_camera_to_map = None
        self.tf_camera_to_base = None
        self.index = 0
        self.p_index = 0

    def __image_callback(self, image_data):
        self.index += 1
        self.image_data = image_data

    def __point_cloud_callback(self, point_cloud_data):
        self.p_index += 1
        self.point_cloud_data = point_cloud_data
        self.listener.waitForTransform("map", self.current_frame, rospy.Time(0), rospy.Duration(1.0))
        self.tf_data = self.listener.lookupTransform("map", self.current_frame, rospy.Time(0))

        self.listener.waitForTransform("base_footprint", self.current_frame, rospy.Time(0), rospy.Duration(1.0))
        self.tf_camera_to_base = self.listener.lookupTransform("base_footprint", self.current_frame, rospy.Time(0))

    def __camera_info_callback(self, camera_info_data):
        self.camera_info_data = camera_info_data

    def __synchronized_callback(self, image_data, point_cloud_data, camera_info_data, cam_base, cam_map):
        self.index += 1
        self.p_index += 1
        self.image_data = image_data
        self.point_cloud_data = point_cloud_data
        self.camera_info_data = camera_info_data

        self.tf_camera_to_base = ([cam_base.pose.pose.position.x,
                                   cam_base.pose.pose.position.y,
                                   cam_base.pose.pose.position.z],
                                  [cam_base.pose.pose.orientation.x,
                                   cam_base.pose.pose.orientation.y,
                                   cam_base.pose.pose.orientation.z,
                                   cam_base.pose.pose.orientation.w])
        self.tf_camera_to_map = ([cam_map.pose.pose.position.x,
                                  cam_map.pose.pose.position.y,
                                  cam_map.pose.pose.position.z],
                                 [cam_map.pose.pose.orientation.x,
                                  cam_map.pose.pose.orientation.y,
                                  cam_map.pose.pose.orientation.z,
                                  cam_map.pose.pose.orientation.w])

    def get_image_array(self):
        if self.image_data is None:
            return None

        height = self.image_data.height
        width = self.image_data.width
        data = self.image_data.data

        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, -1)
        image = np.swapaxes(image, 0, 1)
        return image

    def get_point_cloud(self):
        if self.point_cloud_data is None:
            return None, None, None
        pc = ros_numpy.numpify(self.point_cloud_data)
        points = np.zeros((pc.shape[0] * pc.shape[1], 3))

        points[:, 0] = pc['x'].reshape(-1,)
        points[:, 1] = pc['y'].reshape(-1,)
        points[:, 2] = pc['z'].reshape(-1,)

        non_nan_mask = ~np.isnan(points).any(axis=1)
        non_nan_indices = np.where(non_nan_mask)
        points = points[non_nan_mask]

        return points, self.point_cloud_data, non_nan_indices

    def get_camera_intrinsics(self):
        if self.camera_info_data is None:
            return None

        intrinsics = rs.intrinsics
        intrinsics.coeffs = self.camera_info_data.D
        intrinsics.fx = self.camera_info_data.K[0]
        intrinsics.fy = self.camera_info_data.K[4]
        intrinsics.height = self.camera_info_data.height
        intrinsics.width = self.camera_info_data.width
        intrinsics.model = self.camera_info_data.distortion_model
        intrinsics.ppx = self.camera_info_data.K[2]
        intrinsics.ppy = self.camera_info_data.K[5]

        return intrinsics

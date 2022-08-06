from .point_cloud_classifier_node import PointCloudClassifierNode
from .input_point_cloud_classifier_adapter import InputPointCloudClassifierAdapter
from .point_cloud_classifier_tester import PointCloudClassifierTester
from .output_point_cloud_classifier_adapter import OutputPointCloudClassifierAdapter
from .image_segmentator import ImageSegmentator
from .point_cloud_back_projector import PointCloudBackProjector
from .point_cloud_classifier_accuracy_calculator import PointCloudClassifierAccuracyCalculator
import rospy


class PointCloudClassifierNodeFactory(object):
    def __init__(self, config):
        self.config = config

        camera_parameter_name = rospy.search_param("camera")
        self.camera = rospy.get_param(camera_parameter_name)

    def make_point_cloud_classifier_node(self):
        if self.camera == "first":
            input_point_cloud_classifier_adapter =\
                InputPointCloudClassifierAdapter(image_topic='/d435_1/color/image_raw',
                                                 point_cloud_topic='/d435_1/aligned_points',
                                                 camera_info_topic='/d435_1/color/camera_info',
                                                 camera_base_tf_topic='/cam_base_1',
                                                 camera_map_tf_topic='/cam_map_1',
                                                 current_frame='d435_1_color_optical_frame')
            output_point_cloud_classifier_adapter =\
                OutputPointCloudClassifierAdapter(labeled_points_topic="/labeled_points_1")
        elif self.camera == "second":
            input_point_cloud_classifier_adapter = \
                InputPointCloudClassifierAdapter(image_topic='/d435_2/color/image_raw',
                                                 point_cloud_topic='/d435_2/aligned_points',
                                                 camera_info_topic='/d435_2/color/camera_info',
                                                 camera_base_tf_topic='/cam_base_2',
                                                 camera_map_tf_topic='/cam_map_2',
                                                 current_frame='d435_2_color_optical_frame')
            output_point_cloud_classifier_adapter = \
                OutputPointCloudClassifierAdapter(labeled_points_topic="/labeled_points_2")

        image_segmentator = ImageSegmentator(self.config["test_settings"], self.config["model_settings"])
        point_cloud_back_projector = PointCloudBackProjector()
        point_cloud_classifier_accuracy_calculator = PointCloudClassifierAccuracyCalculator(self.config)

        point_cloud_classifier_tester = PointCloudClassifierTester(image_segmentator,
                                                                   point_cloud_back_projector,
                                                                   point_cloud_classifier_accuracy_calculator,
                                                                   self.config["test_settings"])

        point_cloud_classifier_node = PointCloudClassifierNode(input_point_cloud_classifier_adapter,
                                                               point_cloud_classifier_tester,
                                                               output_point_cloud_classifier_adapter,
                                                               self.config["test_settings"])

        return point_cloud_classifier_node

from PIL import Image
import numpy as np
import os


def save_common_image(mask, index, save_path, name):
    image_name = os.path.join(save_path, name + str(index) + ".png")

    rgb_image_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    rgb_image_array[np.where(mask == 255)] = [255, 255, 255]

    rgb_image = Image.fromarray(rgb_image_array)
    rgb_image.save(image_name)


def save_instance_images(image, index, save_path, name):
    image_name = os.path.join(save_path, name + str(index) + ".png")
    image = Image.fromarray(image)
    image.save(image_name)


class PointCloudClassifierTester(object):
    def __init__(self,
                 image_segmentator,
                 point_cloud_back_projector,
                 point_cloud_classifier_accuracy_calculator,
                 test_settings):
        self.image_segmentator = image_segmentator
        self.point_cloud_back_projector = point_cloud_back_projector
        self.point_cloud_classifier_accuracy_calculator = point_cloud_classifier_accuracy_calculator

        self.test_settings = test_settings

    def launch_point_cloud_classifier_tester(self, image, points, intrinsics, index, first_stream, tf_camera_to_base):
        masks, instance_image, common_predictions = self.image_segmentator.segment_image(image)

        labels = self.point_cloud_back_projector.deproject_point_cloud(masks,
                                                                       points,
                                                                       intrinsics,
                                                                       index,
                                                                       tf_camera_to_base,
                                                                       save_point_cloud_as_image=False)

        if first_stream:
            if self.test_settings["calculate_accuracy"]:
                self.point_cloud_classifier_accuracy_calculator.calculate_accuracy(common_predictions, index)

            if self.test_settings["save_common_output"]:
                if not os.path.exists(self.test_settings['common_output_save_dir']):
                    os.makedirs(self.test_settings['common_output_save_dir'])
                save_common_image(common_predictions,
                                  index,
                                  self.test_settings["common_output_save_dir"],
                                  "collision_avoid_ds_mars")

            if self.test_settings["save_instances_output"]:
                if not os.path.exists(self.test_settings['instance_output_save_dir']):
                    os.makedirs(self.test_settings['instance_output_save_dir'])
                save_instance_images(instance_image,
                                     index,
                                     self.test_settings["instance_output_save_dir"],
                                     "collision_avoid_ds_mars")

        return labels

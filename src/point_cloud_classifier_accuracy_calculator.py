import os
import numpy as np


class PointCloudClassifierAccuracyCalculator(object):
    def __init__(self, config):
        if config["test_settings"]["calculate_accuracy"]:
            self.dir_test = config["test_input_dir"]
            self.label_files = os.listdir(self.dir_test)
            self.labels = [np.load(os.path.join(self.dir_test, label_file))['arr_0'] for label_file in self.label_files]
            self.label_indices = [label_file.replace('collision_avoid_ds_outdoors', '').replace('.npz', '')
                                  for label_file in self.label_files]

    def calculate_accuracy(self, segmented_image, index):
        ground_truth = self.labels[self.label_indices == index]
        correct_predictions = np.where(ground_truth == segmented_image)[0].size

        TP = np.where(ground_truth[np.where(segmented_image == 255)] == 1)[0].size
        FP = np.where(ground_truth[np.where(segmented_image == 255)] == 0)[0].size
        TN = np.where(ground_truth[np.where(segmented_image == 0)] == 0)[0].size
        FN = np.where(ground_truth[np.where(segmented_image == 0)] == 1)[0].size

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        accuracy = correct_predictions / ground_truth.size
        balanced_accuracy = (sensitivity + specificity) / 2
        print("For image {} accuracy is {}, balanced accuracy is {}, stones prediction accuracy is {}".
              format(index, accuracy, balanced_accuracy, sensitivity))

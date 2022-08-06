#!/usr/bin/env python3
import rospy
from src import PointCloudClassifierNodeFactory
from src import PointCloudClassifierTrainer
from src.utils import CropsGenerator
import json
import os

if __name__ == "__main__":
    try:
        print("Loading configuration file")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, 'config.json')) as config:
            config = json.load(config)

            if config["crop_images"]:
                print("Generating crops")
                crops_generator = CropsGenerator(config["images_dir"])
                crops_generator.generate_crops()

            # If train is true - start training
            elif config["training"]:
                print("Starting training")
                point_cloud_classifier_trainer = PointCloudClassifierTrainer(config["train_settings"],
                                                                             config["val_settings"],
                                                                             config["model_settings"])
                point_cloud_classifier_trainer.start_training()

            # Else start testing
            elif config["testing"]:
                rospy.init_node("point_cloud_classifier")
                factory = PointCloudClassifierNodeFactory(config)
                factory.make_point_cloud_classifier_node()
                rospy.spin()

    except rospy.ROSInterruptException:
        pass

import os
import time

import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

import torch
from .stones_dataset import StonesDataset
from .models import get_model
from .utils.utils import Cluster, Visualizer
from .utils import transforms as my_transforms
import numpy as np
from PIL import Image


def save_instance_image(mask, index, i, saver_path):
    image_name = os.path.join(saver_path, "collision_avoid_ds_outdoors" + str(index) + str(i) + ".png")
    rgb_image_array = np.zeros((mask[0], mask[1], 3), dtype=np.uint8)

    rgb_image_array[np.where(mask == 255)] = [255, 255, 255]

    rgb_image = Image.fromarray(rgb_image_array)
    rgb_image.save(image_name)


class ImageSegmentator(object):
    def __init__(self, test_settings, model_settings):
        self.test_settings = test_settings

        if test_settings['display']:
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

        # set device
        device = torch.device("cuda:0" if test_settings['cuda'] else "cpu")

        # load model
        self.model = get_model(model_settings['name'], model_settings['kwargs'])
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load snapshot
        if os.path.exists(test_settings['checkpoint_path']):
            state = torch.load(test_settings['checkpoint_path'])
            self.model.load_state_dict(state['model_state_dict'], strict=True)
        else:
            assert False, 'checkpoint_path {} does not exist!'.format(test_settings['checkpoint_path'])

        self.model.eval()

        # cluster module
        self.cluster = Cluster()

        # Visualizer
        self.visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

    def segment_image(self, image):
        with torch.no_grad():
            image = F.to_tensor(image.copy())
            image = image.unsqueeze(0)

            # make predictions
            output = self.model(image)
            instance_map, predictions = self.cluster.cluster(output[0], threshold=0.9)

            image_size = tuple(self.test_settings["image_size"])
            common_predictions = np.zeros(image_size)
            instance_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            masks = []
            for i, prediction in enumerate(predictions):
                mask = prediction["mask"].cpu().detach().numpy()
                masks.append(mask)
                common_predictions[np.where(mask == 255)] = 255
                instance_image[np.where(mask == 255)] = [(i + 1) * 50, (i + 1) * 50, 0]

            return masks, instance_image, common_predictions

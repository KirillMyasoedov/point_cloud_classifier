import json
import os
import numpy as np
from PIL import Image, ImageDraw
import shutil


json_path = '/home/kirill/project_mars-2022_02_10_18_03_18-datumaro 1.0/dataset/annotations'
instances_saver_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/instances/train'
instances_val_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/instances/val'
rgb_train_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/rgb/train'
rgb_val_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/rgb/val'
labels_path = '/home/kirill/catkin_ws/src/point_cloud_classifier/images/mars/labels'

with open(os.path.join(json_path, 'default.json')) as dataset:
    dataset = json.load(dataset)
    print('Dataset contains the keys', dataset.keys())
    print('info key is', type(dataset['info']))
    print('info contains the keys', dataset['info'].keys())
    print('categories keys is', type(dataset['categories']))
    print('categories contains the keys', dataset['categories'].keys())
    print('categories/label is', type(dataset['categories']['label']))
    print('categories/label contains the keys', dataset['categories']['label'].keys())
    print('categories/label/labels is ', type(dataset['categories']['label']['labels']))
    print('categories/label/labels contains', dataset['categories']['label']['labels'])
    print('categories/label/attributes is', type(dataset['categories']['label']['attributes']))
    print('categories/label/attributes contains', dataset['categories']['label']['attributes'])
    print('items is', type(dataset['items']))
    print('items length is', len(dataset['items']))
    print('items[0] is', type(dataset['items'][0]))
    print('items[0] contains the keys', dataset['items'][0].keys())
    print('items[0]/id is', type(dataset['items'][0]['id']))
    print('items[0]/id contains', dataset['items'][0]['id'])
    print('items[0]/annotations is', type(dataset['items'][0]['annotations']))
    print('items[0]/annotations length is', len(dataset['items'][0]['annotations']))
    print('items[0]/annotations[0] is', type(dataset['items'][0]['annotations'][0]))
    print('items[0]/annotations[0] contains the keys', dataset['items'][0]['annotations'][0].keys())
    print('items[0]/annotations[0]/id contains', dataset['items'][0]['annotations'][0]['id'])
    print('items[0]/annotations[0]/type contains', dataset['items'][0]['annotations'][0]['type'])
    print('items[0]/annotations[0]/attributes contains', dataset['items'][0]['annotations'][0]['attributes'])
    print('items[0]/annotations[0]/group contains', dataset['items'][0]['annotations'][0]['group'])
    print('items[0]/annotations[0]/label_id contains', dataset['items'][0]['annotations'][0]['label_id'])
    print('items[0]/annotations[0]/points contains', dataset['items'][0]['annotations'][0]['points'])
    print('items[0]/annotations[0]/z_order contains', dataset['items'][0]['annotations'][0]['z_order'])
    print('items[0]/attr contains', dataset['items'][0]['attr'])
    print('items[0]/image', dataset['items'][0]['image'])

    for item in dataset['items']:
        common_image_array = np.zeros((640, 480), dtype=np.int32)
        print('Processing image', item['id'])
        image_labels = np.zeros((640, 480))
        for i, annotation in enumerate(item['annotations']):
            image = Image.new('L', (480, 640), 0)
            ImageDraw.Draw(image).polygon(annotation['points'], outline=1, fill=1)
            mask = np.array(image)
            common_image_array += mask * (26000 + i)
            image_labels += mask
            np.savez(os.path.join(labels_path, item['id']), image_labels)

        common_image = Image.fromarray(common_image_array)
        common_image.save(os.path.join(instances_saver_path, item['image']['path']))

    instances_files = os.listdir(instances_saver_path)
    rgb_files = os.listdir(rgb_train_path)
    separation_index = len(instances_files) // 4
    instances_files_val = instances_files[:separation_index]
    rgb_files_val = rgb_files[:separation_index]

    for instance_file, rgb_file in zip(instances_files_val, rgb_files_val):
        shutil.copy(os.path.join(instances_saver_path, instance_file), os.path.join(instances_val_path, instance_file))
        os.remove(os.path.join(instances_saver_path, instance_file))

        shutil.copy(os.path.join(rgb_train_path, rgb_file), os.path.join(rgb_val_path, rgb_file))
        os.remove(os.path.join(rgb_train_path, rgb_file))

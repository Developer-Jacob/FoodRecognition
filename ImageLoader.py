
from collections import defaultdict
from tqdm import tqdm
import time
import os
import configparser
from PIL import Image
import numpy as np

def loadImage(root_path, load_minor_classes):
    print('Load images...')
    total_iamge_count = 0
    since = time.time()
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    boxes = defaultdict()

    for i, major_class in tqdm(enumerate(os.listdir(root_path))):
        major_path = os.path.join(root_path, major_class)
        for index, minor_class in enumerate(os.listdir(major_path)):
            if not(minor_class in load_minor_classes):
                continue
            minor_path = os.path.join(major_path, minor_class)
            file_names = os.listdir(minor_path)
            properties_file = os.path.join(minor_path, 'crop_area.properties')
            boxes = boxDictionary(properties_file)
            for inner_index, file_name in enumerate(file_names):
                if 'jpg' in file_name:
                    class_number = int(file_name.split('_')[1])
                    image_file = Image.open(os.path.join(minor_path, file_name)).convert('RGB')
                    file_name = file_name.split('.')[0]
                    file_name = file_name.lower()

                    if file_name in boxes:
                        image_file = crop(image_file, boxes[file_name])

                    w, h = image_file.size
                    if w > 800:
                        image_file = image_file.resize((w // 2, h // 2), Image.ANTIALIAS)
                    image = np.asarray(image_file)
                    class_number = np.asarray(np.int64(class_number))
                    if isTrainIndex(inner_index, len(file_names)-1):
                        total_iamge_count += 1
                        train_images.append(image)
                        train_labels.append(class_number)
                    else:
                        total_iamge_count += 1
                        test_images.append(image)
                        test_labels.append(class_number)
    time_elapsed = time.time() - since
    print('Finished load image. count: {}, tile: {}'.format(total_iamge_count, time_elapsed))
    return train_images, train_labels, test_images, test_labels

def isTrainIndex(index, total_count):
    return total_count * 0.8 >= index

def crop(image, box):
    if len(box) != 4:
        return image
    box = list(map(int, box))
    return image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))

def boxDictionary(name):
    boxes = defaultdict(list)
    lines = []
    with open(name, 'r') as f:
        lines = f.readlines()
    if not('[sections]\n' in lines):
        lines.insert(0, '[sections]\n')
        with open(name, 'w') as f:
            f.writelines(lines)
    config = configparser.ConfigParser()
    config.read(name)

    for key in config['sections']:
        frame = config['sections'][key].split(',')
        boxes[key] = frame
    return boxes
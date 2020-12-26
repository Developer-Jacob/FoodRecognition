
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
    for i, major_class in enumerate(os.listdir(root_path)):
        if major_class in '.DS_Store':
            continue
        major_path = os.path.join(root_path, major_class)
        for index, minor_class in tqdm(enumerate(os.listdir(major_path)), desc=major_class):
            if minor_class in '.DS_Store':
                continue
            minor_path = os.path.join(major_path, minor_class)
            file_names = os.listdir(minor_path)
            properties_file = os.path.join(minor_path, 'crop_area.properties')
            boxes = boxDictionary(properties_file)
            for inner_index, file_name in enumerate(file_names):
                if 'jpg' in file_name.lower() or 'png' in file_name.lower():
                    class_number = int(file_name.split('_')[1])
                    if not (class_number in load_minor_classes):
                        break
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


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def loadImage2(root_path):
    print('Load images...')
    createFolder('./train')
    createFolder('./test')
    boxes = defaultdict()
    for i, major_class in enumerate(os.listdir(root_path)):
        if major_class in '.DS_Store':
            continue
        major_path = os.path.join(root_path, major_class)
        for index, minor_class in tqdm(enumerate(os.listdir(major_path)), desc=major_class):
            train_folder = './train/{}'.format(minor_class)
            test_folder = './test/{}'.format(minor_class)
            createFolder(train_folder)
            createFolder(test_folder)
            train_images = []
            test_images = []
            if minor_class in '.DS_Store':
                continue
            minor_path = os.path.join(major_path, minor_class)
            file_names = os.listdir(minor_path)
            properties_file = os.path.join(minor_path, 'crop_area.properties')
            boxes = boxDictionary(properties_file)
            for inner_index, file_name in enumerate(file_names):
                if 'jpg' in file_name.lower() or 'png' in file_name.lower():
                    class_number = int(file_name.split('_')[1])
                    image_file = Image.open(os.path.join(minor_path, file_name)).convert('RGB')
                    file_name = file_name.split('.')[0]
                    file_name = file_name.lower()

                    if file_name in boxes:
                        image_file = crop(image_file, boxes[file_name])

                    w, h = image_file.size
                    image = np.asarray(image_file)
                    class_number = np.asarray(np.int64(class_number))
                    if isTrainIndex(inner_index, len(file_names)-1):
                        train_images.append(image)
                    else:
                        test_images.append(image)
            for index, i in enumerate(train_images):
                import torchvision
                i = torchvision.transforms.ToPILImage()(i)
                i.save('{}/{}.jpeg'.format(train_folder, index), 'JPEG')
            for index, i in enumerate(test_images):
                i = torchvision.transforms.ToPILImage()(i)
                i.save('{}/{}.jpeg'.format(test_folder, index), 'JPEG')

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
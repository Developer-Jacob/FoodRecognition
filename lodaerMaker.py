
from collections import defaultdict
from tqdm import tqdm
import time
import os
import configparser
from PIL import Image
import numpy as np
import torch
import torchvision

import transformer
import imageDataSet
def loadImage2(root_path):
    print('Load images...')
    total_image_count = 0

    boxes = defaultdict()
    for i, major_class in enumerate(os.listdir(root_path)):
        if major_class in '.DS_Store':
            continue
        major_path = os.path.join(root_path, major_class)
        for index, minor_class in tqdm(enumerate(os.listdir(major_path)), desc=major_class):
            train_images = []
            train_labels = []
            test_images = []
            test_labels = []

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
                    if w > 800:
                        image_file = image_file.resize((w // 2, h // 2), Image.ANTIALIAS)
                    image = np.asarray(image_file)
                    class_number = np.asarray(np.int64(class_number))
                    if isTrainIndex(inner_index, len(file_names)-1):
                        total_image_count += 1
                        train_images.append(image)
                        train_labels.append(class_number)
                    else:
                        total_image_count += 1
                        test_images.append(image)
                        test_labels.append(class_number)
            train, train_label = transformer.Transformer(train_images, train_labels).getAllImages(transform_list, scales)
            train_set = imageDataSet.ImageDataSet(train, train_label)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=64, shuffle=True)

            test_set = imageDataSet.ImageDataSet(test_images, test_labels, False)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True)
            path = '/Users/nhn/Downloads/kfood/{}.pt'.format(minor_class)
            torch.save({
                'train': train_loader,
                'test': test_loader
            }, path)
    print('Finished load image. count: {}'.format(total_image_count))



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

transform_list = [
    torchvision.transforms.RandomVerticalFlip(1),
    torchvision.transforms.RandomHorizontalFlip(1),
    torchvision.transforms.RandomAffine(30),
    torchvision.transforms.ColorJitter(0.5, 0.5, 0.5)
]

scales = [256]

def imshow(inp, title=None):
    """Imshow for Tensor."""
    import matplotlib.pylab as plt
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

def load(path):
    data = torch.load(path)
    train_loader = data['train']
    test_loader = data['test']

    for i, (inputs, labels) in enumerate(train_loader):
        for index, j in enumerate(range(inputs.size()[0])):
            if index > 5:
                break
            imshow(inputs.cpu().data[j], "Value: {}".format(int(labels[j])))

if __name__ == '__main__':
    # save dictionary = loadImage2('/Users/nhn/Downloads/kfood')
    # load('/Users/nhn/Downloads/kfood/갈비구이.pt')

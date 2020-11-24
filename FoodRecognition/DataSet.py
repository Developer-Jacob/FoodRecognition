import torch
import os
import configparser
from collections import defaultdict
import PIL.Image as pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

ROOT_PATH = "C:/Users/lim/Desktop/kfood"


class ImageDataSet(Dataset):
    def __init__(self, root_path, transforms=None):
        self.transforms = transforms
        self.train_set, self.test_set  = self.loadImage(root_path)

    def loadImage(self, root_path):
        train_images = {}
        test_images = {}
        boxes = defaultdict()
        for major_class in os.listdir(root_path):
            major_path = os.path.join(root_path, major_class)
            major_train_dictionry = {}
            major_test_dictionry = {}
            for minor_class in os.listdir(major_path):
                minor_path = os.path.join(major_path, minor_class)
                minor_train_images = []
                minor_test_images = []
                file_names = os.listdir(minor_path)
                for index, file_name in enumerate(file_names):
                    if 'properties' in file_name:
                        boxes = self.boxDictionary(os.path.join(minor_path, file_name))
                    else:
                        image = pil_image.open(os.path.join(minor_path, file_name))
                        file_name = file_name.split('.')[0]
                        file_name = file_name.lower()
                        if len(boxes[file_name]) != 0:
                            image = self.crop(image, boxes[file_name])

                        if self.isTrainIndex(index, len(file_names)-1):
                            minor_train_images.append(image)
                        else:
                            minor_test_images.append(image)

                major_train_dictionry[minor_class] = minor_train_images
                major_test_dictionry[minor_class] = minor_test_images
            train_images[major_class] = major_train_dictionry
            test_images[major_class] = major_test_dictionry
        return train_images, test_images

    def isTrainIndex(self, index, total_count):
        return total_count * 0.8 >= index

    def __getitem__(self, index):
        return self.transforms(self.train_set[index]), self.transforms(self.test_set[index])

    def __len__(self):
        return len(self.images)

    def showImage(self, images):
        rows = 1
        cols = len(images)
        fig = plt.figure(figsize=(20, 10))
        for index, image in enumerate(images):
            ax = fig.add_subplot(rows, cols, index + 1)
            ax.imshow(image)
            ax.axis("off")
        plt.show()

    def crop(self, image, box):
        box = list(map(int, box))
        if len(box) != 4:
            return image
        return image.crop(tuple(box))

    def boxDictionary(self, name):
        boxes = defaultdict(list)
        with open(name, 'r') as f:
            config_string = '[sections]\n' + f.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        for key in config['sections']:
            frame = config['sections'][key].split(',')
            boxes[key] = frame
        return boxes


if __name__ == "__main__":
    loader = ImageDataSet(ROOT_PATH)
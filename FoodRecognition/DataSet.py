import torch
import os
import configparser
from collections import defaultdict
import PIL.Image as pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

ROOT_PATH = "D:/kfood"
# ROOT_PATH = "C:/Users/lim/Desktop/kfood"

transform = transforms.Compose([transforms.ToPILImage(),
                                # transforms.Resize((256,256)),
                                transforms.ToTensor()])

class ImageDataSet(Dataset):
    def __init__(self, images, labels, isTrain=True):
        self.images = images
        self.labels = labels
        self.isTrain = isTrain

    def __getitem__(self, index):
        return transform(self.images[index]), torch.Tensor(self.labels[index])

    def __len__(self):
        return len(self.images)

class ImageData:
    def __init__(self):
        self.__train_images, self.__train_labels, self.__test_images, self.__test_labels  = self.loadImage(ROOT_PATH)

    def trainDataSet(self):
        return ImageDataSet(self.__train_images, self.__train_labels)

    def testDataSet(self):
        return ImageDataSet(self.__test_images, self.__test_labels, False)

    def loadImage(self, root_path):
        print('Load images...')
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        boxes = defaultdict()
        for major_class in tqdm(os.listdir(root_path)):
            major_path = os.path.join(root_path, major_class)
            for minor_class in tqdm(os.listdir(major_path)):
                minor_path = os.path.join(major_path, minor_class)
                file_names = os.listdir(minor_path)
                for index, file_name in tqdm(enumerate(file_names)):
                    if index > 200:
                        continue
                    if 'properties' in file_name:
                        boxes = self.boxDictionary(os.path.join(minor_path, file_name))
                    elif 'jpg' in file_name:
                        image_file = pil_image.open(os.path.join(minor_path, file_name))
                        file_name = file_name.split('.')[0]
                        file_name = file_name.lower()
                        class_number = int(file_name.split('_')[1])
                        if file_name in boxes != 0:
                            image_file = self.crop(image_file, boxes[file_name])

                        image = np.array(image_file)
                        if self.isTrainIndex(index, len(file_names)-1):
                            train_images.append(image)
                            train_labels.append(class_number)
                        else:
                            test_images.append(image)
                            test_labels.append(class_number)
        print('Finished load image.')
        return train_images, train_labels, test_images, test_labels

    def isTrainIndex(self, index, total_count):
        return total_count * 0.8 >= index

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
        if len(box) != 4:
            return image
        box = list(map(int, box))

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
    print()

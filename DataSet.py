import torch
import os
import configparser
from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time
# vipshome = 'c:\\vips-dev-8.10\\bin'
#
# import os
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']


# LOCAL_ROOT_PATH = "D:/kfood"
LOCAL_ROOT_PATH = "C:/Users/lim/Desktop/kfood"
COLAB_ROOT_PATH = "/content/gdrive/My Drive/kfood"

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                               ])

class ImageDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)

class ImageData:
    def __init__(self, usingColab=True):
        if usingColab:
            try:
                from google.colab import drive
                drive.mount('/content/mnt')
                self.path = COLAB_ROOT_PATH
            except ImportError:
                self.path = LOCAL_ROOT_PATH
        else:
            self.path = LOCAL_ROOT_PATH

        self.__train_images, self.__train_labels, self.__test_images, self.__test_labels = self.loadImage(self.path)

    def trainDataSet(self):
        return ImageDataSet(self.__train_images, self.__train_labels)

    def testDataSet(self):
        return ImageDataSet(self.__test_images, self.__test_labels)

    def loadImage(self, root_path):
        print('Load images...')
        total_iamge_count = 0
        since = time.time()
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        boxes = defaultdict()

        for i, major_class in tqdm(enumerate(os.listdir(root_path))):
            if i > 0:
                break
            major_path = os.path.join(root_path, major_class)
            for index, minor_class in enumerate(os.listdir(major_path)):
                if index > 4:
                    break
                minor_path = os.path.join(major_path, minor_class)
                file_names = os.listdir(minor_path)
                properties_file = os.path.join(minor_path, 'crop_area.properties')
                boxes = self.boxDictionary(properties_file)
                for inner_index, file_name in enumerate(file_names):
                    if 'jpg' in file_name:
                        path = os.path.join(minor_path, file_name)
                        image_file = Image.open(os.path.join(minor_path, file_name)).convert('RGB')
                        # image_file = pyvips.Image.new_from_file(path)
                        file_name = file_name.split('.')[0]
                        file_name = file_name.lower()
                        class_number = int(file_name.split('_')[1])
                        if file_name in boxes:
                            image_file = self.crop(image_file, boxes[file_name])
                        # mem_img = image_file.write_to_memory()
                        # image = np.ndarray(buffer=mem_img, dtype='uint8', shape=[image_file.height, image_file.width, 3])
                        # image = image.reshape(image_file.height, image_file.width, 3)
                        # image_file.draft('RGB', (image_file.height, image_file.width))

                        w, h = image_file.size
                        if w > 1000:
                            image_file = image_file.resize((w // 2, h // 2), Image.ANTIALIAS)

                        image = np.asarray(image_file)
                        # print(image.shape, file_name)
                        class_number = np.asarray(np.int64(class_number))
                        if self.isTrainIndex(inner_index, len(file_names)-1):
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
        return image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))

    def boxDictionary(self, name):
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


if __name__ == "__main__":
    print()

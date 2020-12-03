from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

class ImageDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)


class RandomDataSet(Dataset):
    def __init__(self, images, labels, count):
        print(len(images))
        list = random.sample(range(len(images)), k=count)
        self.images = [images[index] for index in list]
        self.labels = [labels[index] for index in list]

    def __getitem__(self, index):
        return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)
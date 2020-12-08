from torch.utils.data import Dataset
from torchvision import transforms
import random


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor()
                               ])


class ImageDataSet(Dataset):
    def __init__(self, images, labels, isTrain=True):
        self.images = images
        self.labels = labels
        self.isTrain = isTrain

    def __getitem__(self, index):
        if self.isTrain:
            return transforms.ToTensor()(self.images[index]), self.labels[index]
        else:
            return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)


class RandomDataSet(Dataset):
    def __init__(self, images, labels, count):
        random_list = random.sample(range(len(images)), k=count)
        self.images = [images[index] for index in random_list]
        self.labels = [labels[index] for index in random_list]

    def __getitem__(self, index):
        return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)

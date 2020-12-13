from torch.utils.data import Dataset
from torchvision import transforms
import random


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor()
                               ])


class ImageDataSet(Dataset):
    def __init__(self, images, labels, trans, isTrain=True):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.isTrain = isTrain

        if isTrain:
            print('Train image data set count : ', len(self.images))
        else:
            print('Test image data set count : ', len(self.images))

    def __getitem__(self, index):
        if self.isTrain:
            image = self.trans(self.images[index])
            return transforms.ToTensor()(image), self.labels[index]
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

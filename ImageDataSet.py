from torch.utils.data import Dataset
from torchvision import transforms
import random


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor()
                               ])


class ImageDataSet(Dataset):
    def __init__(self, images, labels, input_transforms=[], normalize=None):
        self.images = images
        self.labels = labels
        self.normalize = normalize
        self.input_transforms = input_transforms

    def __getitem__(self, index):
        subIndex = len(self.input_transforms) + 1
        default_image = self.images[index//subIndex]
        image = transforms.ToPILImage()(default_image)
        if index % subIndex != len(self.input_transforms):
            input_transform = self.input_transforms[index % subIndex]
            image = input_transform(image)
        image = transforms.ToTensor()(image)
        image = transforms.Resize((224, 224))(image)

        if self.normalize is not None:
            return self.normalize(image), self.labels[index//subIndex]
        else:
            return image, self.labels[index//subIndex]

    def __len__(self):
        return len(self.images) * (len(self.input_transforms) + 1)


class RandomDataSet(Dataset):
    def __init__(self, images, labels, count):
        random_list = random.sample(range(len(images)), k=count)
        self.images = [images[index] for index in random_list]
        self.labels = [labels[index] for index in random_list]

    def __getitem__(self, index):
        return transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)

from torch.utils.data import Dataset
from torchvision import transforms
import random


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor()
                               ])


class ImageDataSet(Dataset):
    def __init__(self, images, labels, input_transforms, normalize=None):
        self.images = images
        self.labels = labels
        self.normalize = normalize
        self.input_transforms = input_transforms

    def __getitem__(self, index):
        default_image = self.images[index//len(self.input_transforms)]
        default_image = transforms.ToPILImage()(default_image)
        input_transform = self.input_transforms[index % len(self.input_transforms)]
        image = input_transform(default_image)
        image = transforms.ToTensor()(image)
        image = transforms.Resize((224, 224))(image)

        if self.normalize is not None:
            return self.normalize(image), self.labels[index//len(self.input_transforms)]
        else:
            return image, self.labels[index//len(self.input_transforms)]

    def __len__(self):
        if len(self.input_transforms) > 0:
            return len(self.images) * len(self.input_transforms)
        else:
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

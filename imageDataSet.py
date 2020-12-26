from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, images, labels, trans):
        self.images = images
        self.labels = labels
        self.trans = trans

    def __getitem__(self, index):
        return self.trans(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)
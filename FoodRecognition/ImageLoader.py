import torchvision.transforms as transforms

class ImageLoader():
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

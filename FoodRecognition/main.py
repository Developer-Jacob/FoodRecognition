import torch
import torchvision.models as Models
import torch.optim as optim
from torch.utils.data import DataLoader
import FoodRecognition.DataSet as DataSet
from FoodRecognition.Teacher import Student
from FoodRecognition.Teacher import Teacher
import copy
import torch.nn as nn

ROOT_PATH = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

KEY_OPTIMIZER = "optimizer"
KEY_CRITERION = "criterion"
KEY_NETWORKS = "networks"
KEY_EPOCH = "epoch"

class main():
    def __init__(self, params):
        self.optimizer = params[KEY_OPTIMIZER]
        self.criterion = params[KEY_CRITERION]
        self.epoch = params[KEY_EPOCH]
        self.newtworks = params[KEY_NETWORKS]

    def imageLoad(self, batch_size):
        image_data = DataSet.ImageData()
        train_loader = DataLoader(image_data.trainDataSet(), batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)
        return train_loader, test_loader

    def start(self, batch_size):
        train_loader, test_loader = self.imageLoad(batch_size)
        students = []
        for network in self.newtworks:
            students.append(Student(network,
                                    self.epoch,
                                    self.optimizer,
                                    self.criterion, './'))
        teacher = Teacher(students, train_loader, test_loader)
        teacher.teach()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda")

    mobileNet = Models.mobilenet_v2(pretrained=True)
    mobileNet.classifier[1] = nn.Linear(1280, 5)

    network = mobileNet.to(device)

    # for param in network.features.parameters():
    #     param.requires_grad = False

    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    epoch = 50

    params = {KEY_OPTIMIZER: optimizer,
              KEY_CRITERION: criterion,
              KEY_EPOCH: epoch,
              KEY_NETWORKS: [network]
              }

    main = main(params)
    main.start(8)

    
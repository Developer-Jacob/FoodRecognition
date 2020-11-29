import torch
import torchvision.models as Models
import torch.optim as optim
from torch.utils.data import DataLoader
import FoodRecognition.DataSet as DataSet
from FoodRecognition.Teacher import Student
from FoodRecognition.Teacher import Teacher
import torch.nn as nn
import copy

ROOT_PATH = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

KEY_OPTIMIZER = "optimizer"
KEY_CRITERION = "CRITERION"
KEY_NETWORKS = "networks"
KEY_EPOCH = "epoch"

class main():
    def __init__(self, params):
        self.optimizer = params[KEY_OPTIMIZER]
        self.criterion = params[KEY_OPTIMIZER]
        self.epoch = params[KEY_EPOCH]
        self.newtworks = params[KEY_NETWORKS]

    def imageLoad(self, batch_size):
        image_data = DataSet.ImageData()
        train_loader = DataLoader(image_data.trainDataSet(), batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)

    def start(self, loader):
        students = []
        for network in self.newtworks:
            students.append(Student(network,
                                    self.epoch,
                                    copy.deepcopy(self.optimizer),
                                    copy.deepcopy(self.criterion), './'))
        train_loader, test_loader = loader
        teacher = Teacher(students, train_loader, test_loader)
        teacher.teach()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda")


    
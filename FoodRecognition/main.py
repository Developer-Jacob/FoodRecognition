import torch
import torchvision.models as Models
import torch.optim as optim
from torch.utils.data import DataLoader
import FoodRecognition.DataSet as DataSet
from FoodRecognition.Teacher import Student
from FoodRecognition.Teacher import Teacher
import torch.nn as nn

ROOT_PATH = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def net():
    # vgg = Models.vgg16(pretrained=True)
    # vgg.classifier[6] = nn.Linear(4096, 150)
    mobileNet = Models.mobilenet_v2(pretrained=True)
    mobileNet.classifier[1] = nn.Linear(1280, 10)
    return mobileNet

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda")

    image_data = DataSet.ImageData()

    train_loader = DataLoader(image_data.trainDataSet(), batch_size=8, num_workers=0)
    test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)
    net = net()
    # for param in net.features.parameters():
    #     param.requires_grad = False
    network = net.to(device)

    epoch = 300
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    s1 = Student(network, epoch, optimizer, criterion, './')
    students = [s1]
    teacher = Teacher(students, train_loader, test_loader)
    teacher.teach()
    
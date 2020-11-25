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

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda")

    image_data = DataSet.ImageData()
    # print(image_data.testDataSet().__getitem__(0))
    train_loader = DataLoader(image_data.trainDataSet(), batch_size=1, num_workers=0)
    test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)

    # resnet = Models.resnet152(pretrained=True)
    mobileNet = Models.mobilenet_v2(pretrained=True)
    mobileNet.classifier[1] = nn.Linear(1280, 150)
    for param in mobileNet.features.parameters():
        param.requires_grad = False
    mobileNet = mobileNet.to(device)
    print(mobileNet.classifier)
    epoch = 50
    optimizer = optim.Adam(mobileNet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    s1 = Student(mobileNet, epoch, optimizer, criterion, './')
    students = [s1]
    teacher = Teacher(students, train_loader, test_loader)
    teacher.teach()
    
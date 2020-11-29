import torch
import torchvision.models as Models
import torch.optim as optim
import torch.nn as nn

try:
    import FoodRecognition.Recoginizer as rc
    import FoodRecognition.Teacher as tc
except ImportError:
    import Teacher as tc
    import Recognizer as rc

def mobileNet(pretrained):
    mobileNet = Models.mobilenet_v2(pretrained=True)
    mobileNet.classifier[1] = nn.Linear(1280, 5)
    return mobileNet.to(device)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print("Using Cuda")

    # for param in network.features.parameters():
    #     param.requires_grad = False
    epoch = 100

    net = mobileNet(False)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    s1 = tc.Student(net, epoch, optimizer, criterion, './')

    net = mobileNet(True)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    s2 = tc.Student(net, epoch, optimizer, criterion, './')


    recog = rc.Recognizer()
    recog.start([s1, s2], 8)
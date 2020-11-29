import torch
import torchvision.models as Models
import torch.optim as optim
import torch.nn as nn
import Recognizer
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    params = {Recognizer.KEY_OPTIMIZER: optimizer,
              Recognizer.KEY_CRITERION: criterion,
              Recognizer.KEY_EPOCH: epoch,
              Recognizer.KEY_NETWORKS: [network]
              }

    recog = Recognizer(params)
    recog.start(8)

    
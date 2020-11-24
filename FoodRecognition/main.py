import torch
import torchvision.models.resnet as Resnet

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print("Using Cuda")

    resnet = Resnet.resnet152(pretrained=True)
    resnet.to(device)
    print()
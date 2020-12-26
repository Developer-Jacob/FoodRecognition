import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import ImageLoader as loader
import imageDataSet as ds
import Leanring as lr
from torchvision import transforms
import sklearn
import torch.utils.data as data
import Augmentor
import torchvision
import transformer as sv

load_minor_classes = {0: '갈비구이',
                      1: '갈치구이'}

p = Augmentor.Pipeline()
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

trans = transforms.Compose([
    p.torch_transform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

scales = [256]

def loader(trans, path, batch_size, workers):
    folder = torchvision.datasets.ImageFolder(root=path)
    images = [x[0] for x in folder]
    labels = folder.targets
    print(len(images))
    print(len(labels))
    set = ds.ImageDataSet(images, labels, trans)
    return torch.utils.data.DataLoader(set, batch_size=batch_size, num_workers=workers)

if __name__ == "__main__":
    train_loader = loader(trans, './train', 1, 0)
    test_loader = loader(trans, './test', 1, 0)


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
    model_ft.fc = nn.Linear(num_ftrs, 3)

    criterion = nn.CrossEntropyLoss()

    # 모든 매개변수들이 최적화되었는지 관찰
    optimizer_ft = optim.Adam(model_ft.parameters(), 0.001)

    # 7 에폭마다 0.1씩 학습률 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = lr.train_model(model_ft,
                              criterion,
                              optimizer_ft,
                              exp_lr_scheduler,
                              1,
                              train_loader)

    # lr.test_model(model_ft,
    #                    test_loader)


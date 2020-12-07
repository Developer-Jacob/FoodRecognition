import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import ImageLoader as loader
import ImageDataSet as ds
import Leanring as lr
from torchvision import transforms

load_minor_classes = {0: '갈비구이',
                      1: '갈치구이'}

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_list = [
        transforms.RandomVerticalFlip(1),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomAffine(30)
    ]

if __name__ == "__main__":
    train, train_label, test, test_label = loader.loadImage('./kfood', [0, 1])
    train_set = ds.ImageDataSet(train, train_label, transform_list, normalize)
    print('train_image', len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, num_workers=0, shuffle=True)
    test_set = ds.ImageDataSet(test, test_label, [], None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True)
    random_set = ds.RandomDataSet(test, test_label, 3)
    random_loader = torch.utils.data.DataLoader(random_set, batch_size=1, num_workers=0, shuffle=False)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
    model_ft.fc = nn.Linear(num_ftrs, 2)

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
                              train_loader,
                              len(train_set))

    # lr.test_model(model_ft,
    #                    test_loader)
    #
    lr.visualize_model(model_ft,
                    random_loader,
                    load_minor_classes)
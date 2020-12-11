import time
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(network, criterion, optimizer, scheduler, epoch, train_loader, printLog=True):
    network.train()
    network.to(device)
    since = time.time()
    best_acc = 0.0
    best_model = copy.deepcopy(network.state_dict())
    for index, epoch in enumerate(range(epoch)):
        train_loss = 0.0
        train_acc = 0.0
        count = 0
        epoch_since = time.time()
        for inner_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(preds == labels.data)
            count += len(images)
        if scheduler != None:
            scheduler.step()
        epoch_loss = train_loss / count
        epoch_acc = train_acc.double() / count
        if printLog:
            epoch_time_elapsed = time.time() - epoch_since
            print('Epoch: {} Loss: {:.3f}  Acc: {:.4f} Time: {:.0f}m {:.0f}s'
                  .format(index + 1, epoch_loss, epoch_acc, epoch_time_elapsed // 60, epoch_time_elapsed % 60))
        # if epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     best_model = copy.deepcopy(network.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # network.load_state_dict(best_model)
    return network, criterion, optimizer


def test_model(model, test_loader):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        model.train(mode=was_training)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def visualize_model(model, random_loader, class_names):
    model.eval()
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(random_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                prediction = class_names[int(predictions[j])]
                real = class_names[int(labels[j])]
                imshow(inputs.cpu().data[j], "Value: {}, Prediction: {}".format(real, prediction))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

import torch
import time
import torchvision
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Student:
    def __init__(self, network, epoch, optimizer, criterion, path):
        self.network = network
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.path = path
    def save(self):
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.path)
        print('save')

class Teacher:
    def __init__(self, student, train_loader, test_loader):
        self.student = student
        self.train_loader = train_loader
        self.test_loader = test_loader

    def teach(self):
        max_accuracy = 0.0
        self.train(self.student)
        accuracy = self.test(self.student.network)
            # if accuracy > max_accuracy:
                # student.save()

    def train(self, student):
        print('Start training')
        network = student.network
        optimizer = student.optimizer
        criterion = student.criterion
        print(network, criterion, optimizer)
        since = time.time()
        network.train()
        for index, epoch in enumerate(range(student.epoch)):
            train_loss = 0.0
            correct = 0
            total = 0
            count = 0
            for inner_index, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = network(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = outputs.argmax(dim=1, keepdim=True)
                del loss
                del outputs
                count += 1
            time_elapsed = time.time() - since
            print('Epoch: {} Loss: {}  RunningTime: {}m {}s'.format(index+1, train_loss/count, time_elapsed // 60, time_elapsed % 60))


    def test(self, network, num_images=6):
        print('Start testing')
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * correct / total))
        return accuracy

    # def visualize_model(self, model, num_images=6):
    #     was_training = model.training
    #     model.eval()
    #     images_so_far = 0
    #     fig = plt.figure()
    #
    #     with torch.no_grad():
    #         for i, (inputs, labels) in enumerate(self.test_loader):
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #
    #             outputs = model(inputs)
    #             _, preds = torch.max(outputs, 1)
    #
    #             for j in range(inputs.size()[0]):
    #                 images_so_far += 1
    #                 ax = plt.subplot(num_images // 2, 2, images_so_far)
    #                 ax.axis('off')
    #                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
    #                 self.imshow(inputs.cpu().data[j])
    #
    #                 if images_so_far == num_images:
    #                     model.train(mode=was_training)
    #                     return
    #         model.train(mode=was_training)


    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.
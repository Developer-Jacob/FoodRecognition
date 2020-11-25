import torch
import time
import torchvision

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
    def __init__(self, students, train_loader, test_loader):
        self.students = students
        self.train_loader = train_loader
        self.test_loader = test_loader

    def teach(self):
        max_accuracy = 0.0
        for student in self.students:
            self.train(student)
            accuracy = self.test(student.network)
            if accuracy > max_accuracy:
                student.save()

    def train(self, student):
        network = student.network
        optimizer = student.optimizer
        criterion = student.criterion
        since = time.time()
        network.train()
        for index, epoch in enumerate(range(student.epoch)):
            train_loss = 0.0
            correct = 0
            total = 0
            for index, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = network(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            time_elapsed = time.time() - since
            print('Epoch: {} Loss: {}, Accuracy: {}, RunningTime: {}m {}s'.format(index+1, train_loss, (100.0*correct/total), time_elapsed // 60, time_elapsed % 60))


    def test(self, network):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * correct / total))
        return accuracy


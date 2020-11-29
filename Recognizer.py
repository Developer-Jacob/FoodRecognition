from torch.utils.data import DataLoader
import FoodRecognition.DataSet as DataSet
from FoodRecognition.Teacher import Student
from FoodRecognition.Teacher import Teacher

KEY_OPTIMIZER = "optimizer"
KEY_CRITERION = "criterion"
KEY_NETWORKS = "networks"
KEY_EPOCH = "epoch"

class Recognizer():
    def __init__(self, params):
        self.optimizer = params[KEY_OPTIMIZER]
        self.criterion = params[KEY_CRITERION]
        self.epoch = params[KEY_EPOCH]
        self.newtworks = params[KEY_NETWORKS]

    def imageLoad(self, batch_size):
        image_data = DataSet.ImageData()
        train_loader = DataLoader(image_data.trainDataSet(), batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)
        return train_loader, test_loader

    def start(self, batch_size):
        train_loader, test_loader = self.imageLoad(batch_size)
        students = []
        for network in self.newtworks:
            students.append(Student(network,
                                    self.epoch,
                                    self.optimizer,
                                    self.criterion, './'))
        teacher = Teacher(students, train_loader, test_loader)
        teacher.teach()
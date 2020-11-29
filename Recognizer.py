from torch.utils.data import DataLoader

try:
    import FoodRecognition.DataSet as ds
    import FoodRecognition.Teacher as tc
except ImportError:
    import DataSet as ds
    import Teacher as tc

KEY_OPTIMIZER = "optimizer"
KEY_CRITERION = "criterion"
KEY_NETWORKS = "networks"
KEY_EPOCH = "epoch"

class Recognizer():

    def imageLoad(self, batch_size):
        image_data = ds.ImageData()
        train_loader = DataLoader(image_data.trainDataSet(), batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(image_data.testDataSet(), batch_size=1, num_workers=0)
        return train_loader, test_loader

    def start(self, students, batch_size):
        train_loader, test_loader = self.imageLoad(batch_size)

        for student in students:
            teacher = tc.Teacher(student, train_loader, test_loader)
            teacher.teach()
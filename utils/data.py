import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImgDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

class Data():
    def __init__(self, conf):
        self.conf = conf
        self.data_path = conf.data_path
        self.train_loader = None
        self.test_loader = None

    def extract_data(self, kind="train"):
        # you should load image here
        images = np.random.randn(100, 1, 256, 256)
        labels = np.ones(shape=(100))
        return images, labels

    def load_data(self, batch_size=0):
        print("-> load data from: {}".format(self.data_path))
        if not batch_size:
            batch_size = self.conf.batch_size
        x, y = self.extract_data(kind="train")
        self.train_loader = DataLoader(ImgDataset(x, y), batch_size=batch_size, shuffle=True)
        x, y = self.extract_data(kind="test")
        self.test_loader = DataLoader(ImgDataset(x, y), batch_size=batch_size, shuffle=True)
        return self
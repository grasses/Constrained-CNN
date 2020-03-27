import os

class Helper():
    def __init__(self, conf):
        self.conf = conf
        self.init()

    def init(self):
        check_path = [self.conf.model_path, self.conf.data_path]
        for path in check_path:
            if not os.path.exists(path):
                os.makedirs(path)
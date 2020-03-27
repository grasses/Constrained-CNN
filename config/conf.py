import os, torch, random
random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

class Conf(object):
    ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_class = 2
    total_epoch = 10
    batch_size = 1
    learning_rate = 0.001

    data_path = os.path.join(ROOT, "dataset")
    model_path = os.path.join(ROOT, "model")
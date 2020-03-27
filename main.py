import argparse
import numpy as np
import torch
import torch.nn.functional as F
from utils.data import Data
from utils.helper import Helper
from utils.model import MISLnet as Model
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf")
    return parser.parse_args()

def training(conf, model, loader):
    print("\n-> start pretrain model!")
    max_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    for epoch in range(conf.total_epoch):
        for step, (x, y) in enumerate(loader):
            max_step = max(max_step, step)
            global_step = epoch * max(max_step, step) + step
            x, y = x.to(conf.device), y.to(conf.device)
            logist, output = model(x)
            loss = F.cross_entropy(output, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.eq(y.data.view_as(pred)).cpu().sum().item()
            acc = 100.0 * (correct / conf.batch_size)
        print("-> training epoch={:d} loss={:.3f} acc={:.3f}% {:d}".format(epoch, loss, acc, conf.batch_size))

def testing(conf, model, test_loader):
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(conf.device), y.to(conf.device)
            logist, output = model(x)
            test_loss += F.cross_entropy(output, y, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print("-> testing loss={} acc={}".format(test_loss, acc))
    return test_loss, acc

def main():
    args = get_args()
    conf = __import__("config." + args.config, globals(), locals(), ["Conf"]).Conf
    helper = Helper(conf=conf)

    data = Data(conf)
    data.load_data()
    # you need to setup: data.train_loader/data.test_loader

    model = Model(conf).to(conf.device)
    print(model)
    training(conf, model, data.train_loader, helper)

if __name__ == "__main__":
    main()
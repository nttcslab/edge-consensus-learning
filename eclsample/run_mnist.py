# -*- coding: utf-8 -*-
import json
import argparse
import logging
import csv
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from edgecons import GossipSGD
from edgecons import PdmmSGD
from edgecons import AdmmSGD

formatter = '%(asctime)s [ECL] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2)     # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

        # Initialize
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class Kings:
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    train_mask_list = {
        "BALTHASAR": [False, False, False, True, True, True, False, False, False, False],
        "CASPER": [True, True, True, False, False, False, False, False, False, False],
        "MELCHIOR": [False, False, False, False, False, False, True, True, True, True]
    }

    def __init__(self, name, nodes, algorithm="pdmm", device="cpu", interval=6, offset=0, log_dir="/log/"):
        self.logger = logging.getLogger(name)
        self.model = Net().to(device)
        self.device = device

        if algorithm == "gossip":
            self.optimizer = GossipSGD(name, nodes, device, self.model, interval, offset)
        elif algorithm == "admm":
            self.optimizer = AdmmSGD(name, nodes, device, self.model, interval, offset)
        else:  # pdmm_vanilla
            self.optimizer = PdmmSGD(name, nodes, device, self.model, interval, offset)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        self.mask_list = self.train_mask_list[name]
        self.latest_epoch = 0

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        self.log_file_loss = open(log_dir + "/" + name + '_loss.csv', 'w')
        self.writer_loss = csv.writer(self.log_file_loss, lineterminator='\n')

        self.log_file_result = open(log_dir + "/" + name + '_result.csv', 'w')
        self.writer_result = csv.writer(self.log_file_result, lineterminator='\n')

    def train(self, max_epoch=50, batch_size=100, test_interval=10):
        self.logger.info('Training start!!')
        criterion = nn.CrossEntropyLoss()

        # Data Load
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        mask = [self.mask_list[i] for i in train_set.targets]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(mask, len(train_set))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)

        for epoch in range(max_epoch):   # loop over the dataset multiple times
            running_loss = 0.0
            epc_cnt = 0

            self.model.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                # Update
                self.optimizer.update()
                epc_cnt += 1

            self.latest_epoch = epoch + 1
            latest_loss = running_loss / epc_cnt
            diff = self.optimizer.diff()
            latest_diff = diff.item()

            self.model.eval()
            with torch.no_grad():
                test_criterion = nn.CrossEntropyLoss(reduction="sum")
                test_loss = 0.0
                epc_cnt = 0
                for data in self.test_loader:
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = test_criterion(outputs, labels)
                    test_loss += loss.item()
                    epc_cnt += 1

            latest_test_loss = test_loss / epc_cnt / 100
            self.logger.info('[%03d] loss: train %.4f, test %.4f / diff: %.8f' %
                             (self.latest_epoch, latest_loss, latest_test_loss, latest_diff))
            self.writer_loss.writerow([self.latest_epoch, latest_loss, latest_test_loss, latest_diff])

            if self.latest_epoch == 1 or self.latest_epoch % test_interval == 0:
                self.test()

        self.logger.info('Finished Training')
        self.test()

        self.log_file_loss.close()
        self.log_file_result.close()

    def test(self):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))

        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        log_result = []
        log_result.append(self.latest_epoch)

        for i in range(10):
            class_result = 100 * class_correct[i] / class_total[i]
            self.logger.info('Accuracy of %5s : %2d %%' % (Kings.classes[i], class_result))
            log_result.append(class_result)

        self.writer_result.writerow(log_result)


def main():
    parser = argparse.ArgumentParser(description='mnwmnist')
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('-n', '--nodes', required=True)
    parser.add_argument('-a', '--algorithm', default="pdmm")
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = json.load(f)
        name = conf["name"]
        interval = conf["interval"]
        offset = conf["offset"]
        device = conf["device"]

    with open(args.nodes) as f:
        conf = json.load(f)
        nodes = conf["nodes"]

    king = Kings(name, nodes, args.algorithm, device, interval, offset, "log/")
    king.train()


if __name__ == "__main__":
    main()

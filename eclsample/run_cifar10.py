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
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.grp_norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.grp_norm2 = nn.GroupNorm(2, 64)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.grp_norm3 = nn.GroupNorm(2, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.grp_norm4 = nn.GroupNorm(2, 64)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

        # Initialize
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.grp_norm1(self.conv1(x)))
        x = F.relu(self.grp_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)

        x = F.relu(self.grp_norm3(self.conv3(x)))
        x = F.relu(self.grp_norm4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)

        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class Kings:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train_mask_list = {
        "BALTHASAR": [True, True, True, True, True, False, False, True, False, False],
        "CASPER": [True, True, True, True, False, True, False, False, True, False],
        "MELCHIOR": [True, True, True, True, False, False, True, False, False, True]
    }

    def __init__(self, name, nodes, algorithm="pdmm", device="cpu",
                 batch_size=100, interval=6, offset=0, log_dir="/log/"):
        self.logger = logging.getLogger(name)
        self.model = Net().to(device)
        self.device = device
        self.batch_size = batch_size

        if algorithm == "gossip":
            self.optimizer = GossipSGD(name, nodes, device, self.model, interval, offset)
        elif algorithm == "admm":
            self.optimizer = AdmmSGD(name, nodes, device, self.model, interval, offset)
        else:  # pdmm_vanilla
            self.optimizer = PdmmSGD(name, nodes, device, self.model, interval, offset)

        # Data Load
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        mask = [self.train_mask_list[name][i] for i in train_set.targets]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(mask, len(train_set))
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

        self.latest_epoch = 0
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        self.log_file_loss = open(log_dir + "/" + name + '_loss.csv', 'w')
        self.writer_loss = csv.writer(self.log_file_loss, lineterminator='\n')

        self.log_file_result = open(log_dir + "/" + name + '_result.csv', 'w')
        self.writer_result = csv.writer(self.log_file_result, lineterminator='\n')

    def train(self, max_epoch=200, test_interval=10):
        self.logger.info('Training start!!')
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95, last_epoch=100)

        for epoch in range(max_epoch):   # loop over the dataset multiple times
            running_loss = 0.0
            epc_cnt = 0

            self.model.train()
            for data in self.train_loader:
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

            scheduler.step()
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

            latest_test_loss = test_loss / epc_cnt / self.batch_size
            self.logger.info('[%03d] loss: train %.4f, test %.4f / diff: %.8f' %
                             (self.latest_epoch, latest_loss, latest_test_loss, latest_diff))
            self.writer_loss.writerow([self.latest_epoch, latest_loss, latest_test_loss, latest_diff])

            if self.latest_epoch == 1 or self.latest_epoch % test_interval == 0:
                self.test()

        self.logger.info('Finished Training')
        self.test()

    def __del__(self):
        self.log_file_loss.close()
        self.log_file_result.close()

    def test(self):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))

        with torch.no_grad():
            total = 0
            correct = 0
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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        log_result = []
        log_result.append(self.latest_epoch)

        self.logger.info('Accuracy on the 10000 test images: %d %%' % (100 * correct / total))
        for i in range(10):
            class_result = 100 * class_correct[i] / class_total[i]
            self.logger.info('%5s : %2d %%' % (Kings.classes[i], class_result))
            log_result.append(class_result)

        self.writer_result.writerow(log_result)


def main():
    parser = argparse.ArgumentParser(description='mnwcifar10')
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

    king = Kings(name, nodes, args.algorithm, device, 100, interval, offset, "log/")
    king.train()


if __name__ == "__main__":
    main()

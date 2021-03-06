# =========================================================================
# Conventional Convnet Binary with weight normalization for MNIST
# classification
# Copyright (C) <2019>  Georgios Detorakis (gdetor@protonmail.com)
#                       Emre Neftci (eneftci@uci.edu)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =========================================================================
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import Sign_


class cMNIST(nn.Module):
    def __init__(self, batch_size=32, device=0):
        super(cMNIST, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 5)),
                                 name='weight', dim=None)
        self.conv2 = weight_norm(nn.Conv2d(32, 64, kernel_size=(5, 5)),
                                 name='weight', dim=None)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = weight_norm(nn.Linear(64 * 4 * 4, 150),
                               name='weight', dim=None)
        self.fc2 = nn.Linear(150, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        self.sign = Sign_().apply

    def forward(self, x):
        x = self.sign(2*x - 1)
        x = self.sign(self.conv1(x))
        x = self.max_pool(x)
        x = self.sign(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.sign(self.fc1(x))
        x = self.fc2(x)
        return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.max_pool(x)
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = self.max_pool(x)
    #     x = x.view(-1, 64 * 4 * 4)
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     return x


def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    rloss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_ = loss(output, target)
        loss_.backward()
        optimizer.step()
        rloss += loss_.item()
    return rloss / len(train_loader.dataset)


def test(model, device, loss, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
            accuracy += torch.argmax(output, 1).eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    return accuracy, test_loss


if __name__ == '__main__':
    epochs = 100
    batch_size = 100
    test_batch_size = 64
    device = torch.device("cuda:0")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(datasets.MNIST('../data', train=True,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()])),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True, **kwargs)

    test_loader = DataLoader(datasets.MNIST('../data', train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()])),
                             batch_size=test_batch_size,
                             drop_last=True, shuffle=True)

    net = cMNIST(batch_size=batch_size, device=device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0e-6)
    # optimizer = optim.SGD(net.parameters(), lr=0.003,
    #                       momentum=0.9,
    #                       weight_decay=1e-6)
    loss = nn.CrossEntropyLoss()

    rloss = 0
    er_, ac_ = [], []
    for e in range(epochs):
        rloss = train(net, device, train_loader, optimizer, loss, e)
        accur, error = test(net, device, loss, test_loader)
        if e % 2 == 0:
            er_.append(error)
            ac_.append(accur)
            print("Epoch: %d, Loss: %f, Accuracy: %f" % (e, rloss, accur))
    torch.save(net, "mnist_det.pt")
    import sys
    num = sys.argv[1]
    np.save("./data/binary_error_mnist_norm_"+str(num)+"_det", er_)
    np.save("./data/binary_accur_mnist_norm_"+str(num)+"_det", ac_)

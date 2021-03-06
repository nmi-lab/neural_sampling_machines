# =========================================================================
# EMNIST NSM classifier
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
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import NSMLinear_, NSMConv2d_, Sign_


class eMNIST(nn.Module):
    def __init__(self, noise='bernoulli', batch_size=32):
        super(eMNIST, self).__init__()
        input_shape = (28, 28)
        self.fc_in = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.conv1 = NSMConv2d_(1, 32, kernel_size=(5, 5),
                                prob=0.25,
                                bias=True,
                                noise=noise,
                                input_shape=input_shape)
        input_shape = (12, 12)
        self.conv2 = NSMConv2d_(32, 64, kernel_size=(5, 5),
                                prob=0.25,
                                bias=True,
                                noise=noise,
                                input_shape=input_shape)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = NSMLinear_(64 * 4 * 4, 150, prob=0.5, bias=True)
        self.fc2 = nn.Linear(150, 26)
        self.dropout = nn.Dropout(p=0.5)
        self.sign = Sign_().apply
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sign(2*x-1)
        x, _, _ = self.conv1(x)
        x = self.max_pool(x)
        x, _, _ = self.conv2(x)
        x = self.max_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x, _, _ = self.fc1(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    rloss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), (target - 1).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_ = loss(output, target)
        loss_.backward()
        optimizer.step()
        rloss += loss_.item()
    return rloss / len(train_loader.dataset)


def sampling_test(model, device, loss, test_loader, batch_size=16,
                  num_classes=10, num_samples=10):
    model.eval()
    res = torch.zeros((len(test_loader) * batch_size))
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            x, y = data.to(device), (target - 1).to(device)
            out = torch.zeros((batch_size, num_classes)).to(device)
            for n in range(num_samples):
                out += model(x)
            res[i*batch_size:(i+1)*batch_size] = torch.argmax(out, 1).eq(y)
    accuracy = res.mean()
    return 100*accuracy, 100 - 100*accuracy


def test(model, device, loss, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), (target - 1).to(device)
            output = model(data)
            test_loss += loss(output, target).item()
            accuracy += torch.argmax(output, 1).eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    return accuracy, test_loss


if __name__ == '__main__':
    store = True
    epochs = 100
    batch_size = 100
    test_batch_size = 64
    device = torch.device("cuda:0")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    data = datasets.EMNIST('../data', train=True, download=True,
                           split='letters',
                           transform=transforms.Compose([transforms.ToTensor()
                                                         ]))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               **kwargs)

    data = datasets.EMNIST('../data', train=False, split='letters',
                           transform=transforms.Compose([transforms.ToTensor()
                                                         ]))
    test_loader = torch.utils.data.DataLoader(data, batch_size=test_batch_size,
                                              shuffle=True, drop_last=True)

    net = eMNIST(noise='gaussian', batch_size=batch_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0e-6)
    loss = nn.CrossEntropyLoss()

    # rloss, tloss = 0, 0
    ac_, er_ = [], []
    for e in range(epochs):
        rloss = train(net, device, train_loader, optimizer, loss, e)
        # accur, _ = test(net, device, loss, test_loader)
        # print("Epoch: %d, Loss: %f, Accuracy: %f" % (e, rloss, accur))
        if ((e+1) % 50) == 0:
            optimizer.param_groups[-1]['lr'] /= 2
        print("Epoch")
        if(e % 2) == 0:
            ac, er = sampling_test(net, device, loss, test_loader,
                                   batch_size=test_batch_size, num_classes=26,
                                   num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))
    ac_, er_ = np.array(ac_), np.array(er_)
    if store is True:
        import sys
        num = sys.argv[1]
        np.save("./data/gnsm_error_"+str(num)+"_emnist", er_)
        np.save("./data/gnsm_accur_"+str(num)+"_emnist", ac_)

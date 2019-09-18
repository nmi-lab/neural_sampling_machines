# =========================================================================
# Deterministic NSM for MNIST classification (convnet)
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import dNSMLinear_, dNSMConv2d_, Sign_


def hook(module, input, output):
    setattr(module, "_value_hook", output)


class MNIST(nn.Module):
    def __init__(self, noise='bernoulli', xavier_init=False):
        super(MNIST, self).__init__()
        input_shape = (28, 28)
        self.conv1 = dNSMConv2d_(1, 32, kernel_size=(5, 5),
                                 prob=0.25,
                                 bias=True,
                                 input_shape=input_shape,
                                 noise=noise)
        input_shape = (12, 12)
        self.conv2 = dNSMConv2d_(32, 64, kernel_size=(5, 5),
                                 prob=0.25,
                                 bias=True,
                                 noise=noise,
                                 input_shape=input_shape)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = dNSMLinear_(64 * 4 * 4, 512, prob=0.5,
                               bias=True,
                               noise=noise)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.sign = Sign_().apply
        self.tanh = nn.Tanh()

        if xavier_init is True:
            for name in self.named_parameters():
                if 'weight' in name[0]:
                    nn.init.xavier_uniform_(name[1].data)

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
        data, target = data.to(device), target.to(device)
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
            x, y = data.to(device), target.to(device)
            out = torch.zeros((batch_size, num_classes)).to(device)
            for n in range(num_samples):
                out += model(x).softmax(1)
            res[i*batch_size:(i+1)*batch_size] = torch.argmax(out, 1).eq(y)
    accuracy = res.mean()
    return 100*accuracy, 100 - 100*accuracy


if __name__ == '__main__':
    store = True
    var_init = False
    pretrained_w = False
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
                             drop_last=True, shuffle=False)

    net = MNIST(noise='bernoulli').to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-6)
    loss = nn.CrossEntropyLoss()

    rloss, tloss = 0, 0
    ac_, er_ = [], []
    for e in range(epochs):
        rloss = train(net, device, train_loader, optimizer, loss, e)
        if ((e+1) % 50) == 0:
            optimizer.param_groups[-1]['lr'] /= 2
        print("Epoch")
        if(e % 2) == 0:
            ac, er = sampling_test(net, device, loss, test_loader,
                                   batch_size=test_batch_size, num_classes=10,
                                   num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))
    ac_, er_ = np.array(ac_), np.array(er_)
    if store is True:
        import sys
        num = sys.argv[1]
        np.save("./data/nsm_nonstochastic_error_"+str(num)+"_mnist", er_)
        np.save("./data/nsm_nonstochastic_accur_"+str(num)+"_mnist", ac_)

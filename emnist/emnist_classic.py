# =========================================================================
# EMNIST conventional convnet classifier
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


class cMNIST(nn.Module):
    def __init__(self, batch_size=32, device=0):
        super(cMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 150)
        self.fc2 = nn.Linear(150, 26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
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

    net = cMNIST(batch_size=batch_size, device=device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0e-6)
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
    torch.save(net, "emnist_det.pt")
    import sys
    num = sys.argv[1]
    np.save("./data/error_emnist_"+str(num)+"_det", er_)
    np.save("./data/accur_emnist_"+str(num)+"_det", ac_)

# =========================================================================
# Conventional CIFAR10 Allconvnet classifier
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
from tqdm import tqdm
import torch.optim as optim
# from cifar_class import CIFAR10
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.nn.utils import weight_norm
# from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import os
import sys
sys.append.path(os.path.abspath('../model.py'))

from model import Sign_, GaussianNoise


def hook(module, input, output):
    setattr(module, "_value_hook", output)


def get_activities(model, x, dim=2):
    hooks = []
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer'):
            hooks.append(m.register_forward_hook(hook))
    # Compute the activities of the network over one batch
    model.forward(x)
    outs = dict()
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer'):
            outs[n] = (m._value_hook[dim].clone())
    for h in hooks:
        h.remove()
    for i, o in outs.items():
        print('layer {0}, mean: {1:.2f}, std: {2: .2f}'.format(i,
              o.view(batch_size, -1).float().mean(),
              o.view(batch_size, -1).float().std(0).mean()))
    return outs


def init_model(model, x):
    # Add forward hooks to the layers of the network
    hooks = []
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer'):
            hooks.append(m.register_forward_hook(hook))
    # Apply the weights initialization
    for n, m in model.named_modules():
        model.forward(x)
        if hasattr(m, 'nsm_layer') and not hasattr(m, 'is_initd'):
            y = m._value_hook
            out = y
            mu, sigma = out.mean(dim=0), out.std(dim=0)
            m.g.data = 1.0 / sigma

            if m.bias is not None:
                m.bias.data = - mu / sigma
        # Remove all hooks
    for h in hooks:
        h.remove()


N1, N2, N3 = 96, 192, 192


class allconvnet64(nn.Module):
    def __init__(self):
        super(allconvnet64, self).__init__()
        self.sign = Sign_().apply
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(.5)
        self.apool = nn.AvgPool2d(kernel_size=8)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.tanh = nn.Tanh()

        self.noise = GaussianNoise(sigma=0.15)
        self.b1 = nn.BatchNorm2d(N1)

        self.input = weight_norm(nn.Conv2d(3, N1, kernel_size=(3, 3),
                                           padding=(1, 1)),
                                 name='weight', dim=None)
        self.convset1 = nn.ModuleList([
            weight_norm(nn.Conv2d(N1, N1,  kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None),
            weight_norm(nn.Conv2d(N1, N1, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None),
            weight_norm(nn.Conv2d(N1, N1, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None)]
            )
        self.convset2 = nn.ModuleList([
            weight_norm(nn.Conv2d(N1, N2, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None),
            weight_norm(nn.Conv2d(N2, N2, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None),
            weight_norm(nn.Conv2d(N2, N2, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight', dim=None)]
            )
        self.convset3 = nn.ModuleList([
            weight_norm(nn.Conv2d(N2, N3, kernel_size=(3, 3), padding=[0, 0]),
                        name='weight', dim=None)])
        self.nin1 = weight_norm(nn.Conv2d(N3, N3, kernel_size=(1, 1),
                                padding=[0, 0]), name='weight', dim=None)
        self.nin2 = weight_norm(nn.Conv2d(N3, N3, kernel_size=(1, 1),
                                padding=[0, 0]), name='weight', dim=None)
        self.fc_out = weight_norm(nn.Linear(in_features=N3, out_features=10),
                                  name='weight', dim=None)

        # for name in self.named_parameters():
        #     print(name[0])
        #     if 'weight' in name[0]:
        #         # name[1].data.normal_(0, 0.05)
        #     if 'bias' in name[0]:
        #         name[1].data.zero_()

    def forward(self, x):
        x = self.noise(x)
        x = self.input(x)
        x = self.b1(x)
        # x = self.sign(x)
        for i, layer in enumerate(self.convset1):
            x = self.relu(layer(x))
        x = self.mpool(x)
        x = self.dropout(x)
        for i, layer in enumerate(self.convset2):
            x = self.relu(layer(x))
        x = self.mpool(x)
        x = self.dropout(x)
        for i, layer in enumerate(self.convset3):
            x = self.relu(layer(x))
        x = self.relu(self.nin1(x))
        x = self.relu(self.nin2(x))
        x = self.apool(x)
        x = x.view(-1, N3)
        x = self.fc_out(x)
        return x


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
    return 100 * accuracy, 100 - 100 * accuracy


def sampling_test(model, device, loss, loader,  num_classes=11, num_samples=4):
    isnsm = False
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer'):
            isnsm = True
    if not isnsm:
        model.eval()
    # else:
    #     model.train()
    batch_size = loader.batch_size
    res = torch.zeros((len(loader) * batch_size))
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(loader)):
            x, y = data.to(device), target.to(device)
            out = torch.zeros((batch_size, num_classes)).to(device)
            k = 0
            while True:
                idx = torch.arange(0, batch_size).long()
                if k > num_samples:
                    idx = idx[out[idx].softmax(1).max(1)[0] < .9]
                if len(idx) == 0 or k > 100:
                    break
                else:
                    out[idx] += model(x[idx]).softmax(1)
                    k += 1

            res[i*batch_size:(i+1)*batch_size] = torch.argmax(out, 1).eq(y)
    accuracy = res.mean()
    return 100*accuracy, 100 - 100*accuracy, res


if __name__ == '__main__':
    epochs = 200
    batch_size = 100
    test_batch_size = 100
    device = torch.device("cuda:0")

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010)),])
    zca_train = torch.from_numpy(np.load("zca_mat_train_1.npy").astype('f'))
    zca_test = torch.from_numpy(np.load("zca_mat_test_1.npy").astype('f'))

    test_loader = DataLoader(CIFAR10('/share/data', train=False,
                                     transform=transform_test),
                                     # transform=transforms.Compose([
                                     #            transforms.ToTensor(),
                                     #            normalize,
                                     #            transforms.LinearTransformation(zca_test)
                                     #            ])),
                             batch_size=test_batch_size,
                             drop_last=True, shuffle=False,
                             num_workers=5, pin_memory=True)

    train_loader = DataLoader(CIFAR10('/share/data', train=True,
                                      transform=transform_train),
                                      # transform=transforms.Compose([
                                      #           transforms.ToTensor(),
                                      #           normalize,
                                      #           transforms.LinearTransformation(zca_train)
                                      #           ])),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=1,
                              pin_memory=True)

    net = allconvnet64().to(device)
    # net = torch.load("cifar10_trained_model_full_noinputcnn.pt").to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    # data = iter(train_loader).next()
    # x, y = data[0].to(device), data[1].to(device)
    # init_model(net, x.to(device))
    # outs = get_activities(net, x.to(device))

    ll, mm = len(train_loader), len(test_loader)
    ac_, er_ = [], []

    for e in range(epochs):
        rloss = 0.0
        optimizer.param_groups[-1]['lr'] = (0.001 * np.minimum(2. - e/100., 1.))
        print("Epoch: %d, LRate: %f" % (e, optimizer.param_groups[-1]['lr']))
        print("Betas: ", optimizer.param_groups[-1]['betas'])
        if e == 100:
            optimizer.param_groups[-1]['betas'] = (0.5, 0.999)
        for i, data in (enumerate(train_loader)):
            # Get the data
            x, y = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # Run the forward pass
            y_ = net(x)

            # Run the backward pass
            loss_ = loss(y_, y)
            loss_.backward()
            optimizer.step()

            # Keep some scores
            rloss += loss_.item()
            lr = optimizer.param_groups[-1]['lr']
            print(loss_, lr, end='\r')
        print("Loss: {0:.4f}".format(rloss))
        if((e+1) % 5) == 0:
            ac, er, res = sampling_test(net, device, loss, test_loader,
                                        num_classes=10, num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))

        print("Epoch: %d, Loss: %f" % (e, rloss/ll))
    torch.save(net, "cifar10_trained_model_full_noinputcnn.pt")
    ac_, er_ = np.array(ac_), np.array(er_)
    np.save("./det_error_cifar10_3", er_)
    np.save("./det_accur_cifar10_3", ac_)

# =========================================================================
# CIFAR100 NSM Allconvnet classifier
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
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.nn import functional as F
# from cifar100_new_classic import allconvnet64
from torch.utils.data.dataloader import DataLoader
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import NSMConv2d_, Sign_, GaussianNoise


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

    # Compute the activities of the network over one batch
    model.forward(x)

    # Apply the weights initialization
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer') and not hasattr(m, 'is_initd'):
            y = m._value_hook
            m.weight.data *= 5
            out = y[2]
            mu, sigma = out.mean(dim=0), out.std(dim=0)
            # sigma[sigma < 1e-1] = 1e-1
            s = sigma
            m.b.data = m.b.data / s
            m.is_initd = True

            if len(m.weight.data.shape) > 2:
                wn = m.weight.data
                wnn = wn.view(-1, wn.shape[0])
                k2 = F.normalize(wnn, p=2, dim=0)
                k2 = torch.norm(k2, dim=0)
            else:
                k2 = torch.norm(m.weight.data, dim=1)

            if m.bias is not None:
                mmu = mu.view(mu.shape[0], -1).mean(dim=1)
                ms = s.view(s.shape[0], -1).mean(dim=1)
                m.bias.data = -(mmu * m.ap * k2) / ms
            break

        # Remove all hooks
    for h in hooks:
        h.remove()


N1, N2, N3 = 96, 192, 192


class allnsmconvnet64(nn.Module):
    def __init__(self, noise='bernoulli'):
        super(allnsmconvnet64, self).__init__()
        self.sign = Sign_().apply
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apool = nn.AvgPool2d(kernel_size=8)

        self.noise = GaussianNoise(sigma=0.15)
        self.input = nn.Conv2d(3, N1, kernel_size=(3, 3), padding=(1, 1))
        self.b0 = nn.BatchNorm2d(3)
        self.b1 = nn.BatchNorm2d(N1)
        self.b2 = nn.BatchNorm1d(N3)

        self.convset1 = nn.ModuleList([
            NSMConv2d_(N1, N1,  kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=False),
            NSMConv2d_(N1, N1, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=False),
            NSMConv2d_(N1, N1, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=False)]
            )
        self.convset2 = nn.ModuleList([
            NSMConv2d_(N1, N2, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=False),
            NSMConv2d_(N2, N2, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=False),
            NSMConv2d_(N2, N2, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=False)]
            )
        self.convset3 = nn.ModuleList([
            NSMConv2d_(N2, N3, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[8, 8], noise=noise, bias=False),
            NSMConv2d_(N3, N3, kernel_size=(1, 1), prob=.5, padding=[0, 0],
                       input_shape=[8, 8], noise=noise, bias=False),
            NSMConv2d_(N3, N3, kernel_size=(1, 1), prob=.5, padding=[0, 0],
                       input_shape=[8, 8], noise=noise, bias=False)]
            )
        self.fc_out = nn.Linear(in_features=N3, out_features=100)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.noise(x)
        x = self.b0(x)
        x = self.input(x)
        x = self.b1(x)
        # x = self.sign(2*x-1)
        for i, layer in enumerate(self.convset1):
            x = layer(x)[0]
        x = self.mpool(x)
        for i, layer in enumerate(self.convset2):
            x = layer(x)[0]
        x = self.mpool(x)
        for i, layer in enumerate(self.convset3):
            x = layer(x)[0]
        x = self.apool(x)
        x = x.view(-1, N3)
        x = self.b2(x)
        x = self.fc_out(x)
        return x


def cifar10_replace_weights(net1, net2):
    w1 = net1.convset1[0].weight_g * (net1.convset1[0].weight_v /
                                      net1.convset1[0].weight_v.norm())
    w2 = net1.convset1[1].weight_g * (net1.convset1[1].weight_v /
                                      net1.convset1[1].weight_v.norm())
    w3 = net1.convset1[2].weight_g * (net1.convset1[2].weight_v /
                                      net1.convset1[2].weight_v.norm())

    w4 = net1.convset2[0].weight_g * (net1.convset2[0].weight_v /
                                      net1.convset2[0].weight_v.norm())
    w5 = net1.convset2[1].weight_g * (net1.convset2[1].weight_v /
                                      net1.convset2[1].weight_v.norm())
    w6 = net1.convset2[2].weight_g * (net1.convset2[2].weight_v /
                                      net1.convset2[2].weight_v.norm())

    w7 = net1.convset3[0].weight_g * (net1.convset3[0].weight_v /
                                      net1.convset3[0].weight_v.norm())
    w8 = net1.nin1.weight_g * (net1.nin1.weight_v /
                               net1.nin1.weight_v.norm())
    w9 = net1.nin2.weight_g * (net1.nin2.weight_v /
                               net1.nin2.weight_v.norm())

    # w1 = net1.convset1[0].weight
    # w2 = net1.convset1[1].weight
    # w3 = net1.convset1[2].weight

    # w4 = net1.convset2[0].weight
    # w5 = net1.convset2[1].weight
    # w6 = net1.convset2[2].weight

    # w7 = net1.convset3[0].weight
    # w8 = net1.convset3[1].weight
    # w9 = net1.convset3[2].weight

    # w1 = net1.convset1[0].weight_v
    # w2 = net1.convset1[1].weight_v
    # w3 = net1.convset1[2].weight_v

    # w4 = net1.convset2[0].weight_v
    # w5 = net1.convset2[1].weight_v
    # w6 = net1.convset2[2].weight_v

    # w7 = net1.convset3[0].weight_v
    # w8 = net1.nin1.weight_v
    # w9 = net1.nin2.weight_v

    net2.input.weight.data = net1.input.weight.data
    net2.convset1[0].weight.data = w1
    net2.convset1[1].weight.data = w2
    net2.convset1[2].weight.data = w3

    net2.convset2[0].weight.data = w4
    net2.convset2[1].weight.data = w5
    net2.convset2[2].weight.data = w6

    net2.convset3[0].weight.data = w7
    net2.convset3[1].weight.data = w8
    net2.convset3[2].weight.data = w9

    net2.fc_out.weight.data = net1.fc_out.weight.data
    net2.fc_out.bias.data = net1.fc_out.bias.data


def replace_init_weights(net1, net2):
    w, b = [], []
    for name in net1.named_parameters():
        if 'weight' in name[0]:
            print(name[0], name[1].shape)
            w.append(name[1].data)
        if 'bias' in name[0]:
            b.append(name[1].data)

    i = 0
    for name in net2.named_parameters():
        if 'weight' in name[0]:
            print(name[0], name[1].shape)
            name[1].data = w[i]
            i += 1

    i = 0
    for name in net2.named_parameters():
        if 'bias' in name[0]:
            name[1].data = b[i]
            i += 1


def sampling_test(model, device, loss, loader,  num_classes=11, num_samples=4):
    isnsm = False
    for n, m in model.named_modules():
        if hasattr(m, 'nsm_layer'):
            isnsm = True
    if not isnsm:
        model.eval()
    else:
        model.train()
    batch_size = loader.batch_size
    res = torch.zeros((len(loader) * batch_size))
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(loader)):
            # x, y = data.to(device).half(), target.to(device)
            # out = torch.zeros((batch_size, num_classes)).to(device).half()
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
    epochs = 300
    batch_size = 100
    test_batch_size = 100
    device = torch.device("cuda:0")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))
                                         ])

    test_loader = DataLoader(CIFAR100('/share/data', train=False,
                                     transform=transform_test),
                             batch_size=test_batch_size,
                             drop_last=True, shuffle=False,
                             num_workers=5, pin_memory=True)

    train_loader = DataLoader(CIFAR100('/share/data', train=True,
                                      transform=transform_train),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=5,
                              pin_memory=True)

    net = allnsmconvnet64(noise='bernoulli').to(device)
    # net_ = torch.load("cifar10_trained_model_full_noinputcnn.pt").to(device)
    # cifar10_replace_weights(net_, net)

    # x, y = next(iter(train_loader))
    # for _ in range(20):
    #     init_model(net, x.to(device))
    # outs = get_activities(net, x.to(device))

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    ll, mm = len(train_loader), len(test_loader)
    ac_, er_ = [], []

    for e in range(epochs):
        rloss = 0.0
        # if ((e+1) % 30) == 0:
        #     optimizer.param_groups[-1]['lr'] /= 2
        # optimizer.param_groups[-1]['lr'] = (0.001 * np.minimum(2. - e/100., 1.))
        optimizer.param_groups[-1]['lr'] = (0.001 * np.minimum(1.5 - e*0.005, 1.))
        # print("Epoch: %d, LRate: %f" % (e, optimizer.param_groups[-1]['lr']))
        if e == 100:
            optimizer.param_groups[-1]['betas'] = (0.45, 0.999)
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
            print(loss_, end='\r')
        print("Loss {0:.4f}".format(rloss))

        if((e+1) % 5) == 0:
            ac, er, res = sampling_test(net, device, loss, test_loader,
                                        num_classes=100, num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))

        print("Epoch: %d, Loss: %f" % (e, rloss/ll))
    torch.save(net, "cifar100_trained_model_full_300_clean.pt")

    ac_, er_ = np.array(ac_), np.array(er_)
    np.save("./data/bnsm_error_cifar100_300_clean", er_)
    np.save("./data/bnsm_accur_cifar100_300_clean", ac_)

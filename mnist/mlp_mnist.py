# =========================================================================
# MLP (linear) NSM for MNIST classification
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import NSMLinear_, Sign_


def hook(module, input, output):
    setattr(module, "_value_hook", output)


class mnist_(nn.Module):
    def __init__(self, noise='bernoulli', xavier_init=False):
        super(mnist_, self).__init__()
        self.fc1 = NSMLinear_(784, 300, prob=0.5,
                              bias=True,
                              noise=noise)
        self.fc2 = NSMLinear_(300, 300, prob=0.5,
                              bias=True,
                              noise=noise)
        self.fc3 = NSMLinear_(300, 300, prob=0.5,
                              bias=True,
                              noise=noise)
        self.fc4 = nn.Linear(300, 10)
        self.sign = Sign_().apply
        self.tanh = nn.Tanh()

        if xavier_init is True:
            for name in self.named_parameters():
                if 'weight' in name[0]:
                    nn.init.xavier_uniform_(name[1].data)

    def forward(self, x):
        x = self.sign(2*x-1)
        x = x.view(-1, 28*28)
        x, _, _ = self.fc1(x)
        x, _, _ = self.fc2(x)
        x, _, _ = self.fc3(x)
        x = self.fc4(x)
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


def replace_init_weights(net1, net2):
    w, b = [], []
    for name in net1.named_parameters():
        if 'weight' in name[0]:
            w.append(name[1].data)
        if 'bias' in name[0]:
            b.append(name[1].data)

    i = 0
    for name in net2.named_parameters():
        if 'weight' in name[0]:
            name[1].data = w[i]
            i += 1

    i = 0
    for name in net2.named_parameters():
        if 'bias' in name[0]:
            name[1].data = b[i]
            i += 1


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
            # out = y[1]
            mu, sigma = out.mean(dim=0), out.std(dim=0)
            # sigma[sigma < 1e-1] = 1e-1
            s = sigma
            print(s.max(), s.min())
            # m.b.data = m.b.data / s
            m.b.data = 1.0 / (s + 1e-6)
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
    epochs = 400
    batch_size = 100
    test_batch_size = 100
    device = torch.device("cuda:0")

    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    kwargs = {'num_workers': 5, 'pin_memory': True}
    train_loader = DataLoader(datasets.MNIST('../data', train=True,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 ])),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True, **kwargs)

    test_loader = DataLoader(datasets.MNIST('../data', train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                ])),
                             batch_size=test_batch_size,
                             drop_last=True, shuffle=False, **kwargs)

    net = mnist_(noise='bernoulli', xavier_init=True).to(device)
    net_ = torch.load("mlp_mnist.pt")
    replace_init_weights(net_, net)
    optimizer = optim.Adam(net.parameters(), lr=0.00031)
    loss = nn.CrossEntropyLoss()

    # x, y = next(iter(train_loader))
    # for _ in range(4):
    #     init_model(net, x.to(device))
    # outs = get_activities(net, x.to(device))

    if pretrained_w is True:
        replace_init_weights(net)

    rloss, tloss = 0, 0
    ac_, er_ = [], []
    for e in range(epochs):
        # optimizer.param_groups[-1]['lr'] = (0.001 * np.minimum(2. - e/100., 1.))
        if ((e+1) % 50) == 0:
            optimizer.param_groups[-1]['lr'] /= 2
        # if e == 100:
        #     optimizer.param_groups[-1]['betas'] = (0.5, 0.999)
        rloss = train(net, device, train_loader, optimizer, loss, e)
        print("Epoch: %d, LRate: %f" % (e, optimizer.param_groups[-1]['lr']))
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
        np.save("./data/mlp_bnsm_error_"+str(num)+"_mnist", er_)
        np.save("./data/mlp_bnsm_accur_"+str(num)+"_mnist", ac_)

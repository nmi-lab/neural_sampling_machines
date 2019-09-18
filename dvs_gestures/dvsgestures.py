# =========================================================================
# DVS Gestures Allconvnet NSM
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
from torch import nn
import torch.optim as optim
from load_dvsgestures_sparse import DVSGestures
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import os
import sys
sys.append.path(os.path.abspath('../model.py'))
from model import NSMConv2d_, Sign_


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
            out = y[0]
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


N1, N2, N3, N4 = 64, 96, 128, 256


class allnsmconvnet64(nn.Module):
    def __init__(self, noise='bernoulli'):
        super(allnsmconvnet64, self).__init__()
        self.sign = Sign_().apply
        self.mpool = nn.MaxPool2d(kernel_size=2)
        self.apool = nn.AvgPool2d(kernel_size=8)
        self.convset1 = nn.ModuleList([
            NSMConv2d_(6, N1,  kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[64, 64], noise=noise, bias=True),
            NSMConv2d_(N1, N1, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[64, 64], noise=noise, bias=True),
            NSMConv2d_(N1, N1, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[64, 64], noise=noise, bias=True)]
            )
        self.convset2 = nn.ModuleList([
            NSMConv2d_(N1, N2,  kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=True),
            NSMConv2d_(N2, N2, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=True),
            NSMConv2d_(N2, N2, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[32, 32], noise=noise, bias=True)]
            )
        self.convset3 = nn.ModuleList([
            NSMConv2d_(N2, N3, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=True),
            NSMConv2d_(N3, N3, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=True),
            NSMConv2d_(N3, N3, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[16, 16], noise=noise, bias=True)]
            )
        self.convset4 = nn.ModuleList([
            NSMConv2d_(N3, N4, kernel_size=(3, 3), prob=.5, padding=[1, 1],
                       input_shape=[8, 8], noise=noise, bias=True),
            NSMConv2d_(N4, N4, kernel_size=(1, 1), prob=.5, padding=[0, 0],
                       input_shape=[8, 8], noise=noise, bias=True),
            NSMConv2d_(N4, N4, kernel_size=(1, 1), prob=.5, padding=[0, 0],
                       input_shape=[8, 8], noise=noise, bias=True)]
            )
        self.fc_out = nn.Linear(in_features=N4, out_features=11)

        # for name in self.named_parameters():
        #     if 'weight' in name[0]:
        #         nn.init.xavier_uniform_(name[1].data)

    def forward(self, x):
        x = self.sign(2*x-1)
        for i, layer in enumerate(self.convset1):
            x = layer(x)[0]
        x = self.mpool(x)
        for i, layer in enumerate(self.convset2):
            x = layer(x)[0]
        x = self.mpool(x)
        for i, layer in enumerate(self.convset3):
            x = layer(x)[0]
        x = self.mpool(x)
        for i, layer in enumerate(self.convset4):
            x = layer(x)[0]
        x = self.apool(x)
        x = x.view(-1, N4)
        x = self.fc_out(x)
        return x


def dvs_gestures_replace_w(net1, net2):
    # for name in net1.named_parameters():
    #     if 'weight' in name[0]:
    #         print(name[0], name[1].shape)

    j = 0
    for i in range(len(net2.convset1)):
        net2.convset1[i].weight.data = net1.convset1[i].weight_v.data
        # net2.convset1[i].bias.data = net1.convset1[j].bias.data
        j += 3

    j = 0
    for i in range(len(net2.convset2)):
        net2.convset2[i].weight.data = net1.convset2[i].weight_v.data
        # net2.convset2[i].bias.data = net1.convset2[j].bias.data
        j += 3

    j = 0
    for i in range(len(net2.convset3)):
        net2.convset3[i].weight.data = net1.convset3[i].weight_v.data
        # net2.convset3[i].bias.data = net1.convset3[j].bias.data
        j += 3

    j = 0
    for i in range(len(net2.convset4)):
        net2.convset4[i].weight.data = net1.convset4[i].weight_v.data
        # net2.convset4[i].bias.data = net1.convset4[j].bias.data
        j += 3

    net2.fc_out.weight.data = net1.fc_out.weight.data
    net2.fc_out.bias.data = net1.fc_out.bias.data


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
    test_batch_size = 64
    device = torch.device("cuda:0")

    train_filename = "/share/data/DvsGesture/dvs_gestures_dense_train.pt"
    test_filename = "/share/data/DvsGesture/dvs_gestures_dense_test.pt"

    train_loader = DataLoader(DVSGestures(train_filename),
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(DVSGestures(test_filename),
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=True)

    net = allnsmconvnet64(noise='gaussian').to(device)
    from dvsgestures_classic import allconvnet64
    net_ = allconvnet64().to(device)
    net_ = torch.load("dvs_gestures_trained_model_full_new.pt")
    dvs_gestures_replace_w(net_, net)

    # x, y = next(iter(train_loader))
    # init_model(net, x.to(device))
    # outs = get_activities(net, x.to(device))

    optimizer = optim.Adam(net.parameters(), lr=0.0003, eps=1e-4)
    loss = nn.CrossEntropyLoss()

    ll, mm = len(train_loader), len(test_loader)
    ac_, er_ = [], []

    for e in range(epochs):
        rloss = 0.0
        if ((e+1) % 100) == 0:
            optimizer.param_groups[-1]['lr'] /= 2
        print("Epoch", e)
        for i, data in (enumerate(train_loader)):
            x, y = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            y_ = net(x)

            loss_ = loss(y_, y)
            loss_.backward()
            optimizer.step()

            rloss += loss_.item()
            print(loss_, end='\r')
        print("Loss {0:.4f}".format(rloss))

        if((e+1) % 5) == 0:
            ac, er, res = sampling_test(net, device, loss, test_loader,
                                        num_classes=11, num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))

        print("Epoch: %d, Loss: %f" % (e, rloss/ll))
    torch.save(net, "dvs_gestures_trained_model_full.pt")

    import numpy as np
    ac_, er_ = np.array(ac_), np.array(er_)
    np.save("./data/gansm_error_1_dvs", er_)
    np.save("./data/gansm_accur_1_dvs", ac_)

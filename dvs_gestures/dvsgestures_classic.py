# =========================================================================
# DVS Gestures Allconvnet conventional network
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
from torch.nn.utils import weight_norm
from tqdm import tqdm
from model import Sign_
import torch.optim as optim
from load_dvsgestures_sparse import DVSGestures
from torch.utils.data.dataloader import DataLoader


N1, N2, N3, N4 = 64, 96, 128, 256


class allconvnet64(nn.Module):
    def __init__(self):
        super(allconvnet64, self).__init__()
        self.sign = Sign_().apply
        self.mpool = nn.MaxPool2d(kernel_size=2)
        self.apool = nn.AvgPool2d(kernel_size=8)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.convset1 = nn.ModuleList([
            weight_norm(nn.Conv2d(6,  N1,  kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N1),
            weight_norm(nn.Conv2d(N1, N1, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N1),
            weight_norm(nn.Conv2d(N1, N1, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.5)]
            )
        self.convset2 = nn.ModuleList([
            weight_norm(nn.Conv2d(N1, N2,  kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N2),
            weight_norm(nn.Conv2d(N2, N2, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N2),
            weight_norm(nn.Conv2d(N2, N2, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.5)]
            )
        self.convset3 = nn.ModuleList([
            weight_norm(nn.Conv2d(N2, N3, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N3),
            weight_norm(nn.Conv2d(N3, N3, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N3),
            weight_norm(nn.Conv2d(N3, N3, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.5)]
            )
        self.convset4 = nn.ModuleList([
            weight_norm(nn.Conv2d(N3, N4, kernel_size=(3, 3), padding=[1, 1]),
                        name='weight'),
            # nn.BatchNorm2d(N4),
            weight_norm(nn.Conv2d(N4, N4, kernel_size=(1, 1), padding=[0, 0]),
                        name='weight'),
            # nn.BatchNorm2d(N4),
            weight_norm(nn.Conv2d(N4, N4, kernel_size=(1, 1), padding=[0, 0]),
                        name='weight')]
            )
        self.fc_out = nn.Linear(in_features=N4, out_features=11)
        # self.bn_in = nn.BatchNorm2d(6)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x):
        # x = self.sign(2*x-1)
        # x = self.bn_in(x)
        for i, layer in enumerate(self.convset1):
            x = self.relu(layer(x))
        for i, layer in enumerate(self.convset2):
            x = self.relu(layer(x))
        for i, layer in enumerate(self.convset3):
            x = self.relu(layer(x))
        for i, layer in enumerate(self.convset4):
            x = self.relu(layer(x))
        x = self.apool(x)
        x = self.bn(x)
        x = x.view(-1, N4)
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
    return accuracy, test_loss


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

    # kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(DVSGestures(train_filename),
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(DVSGestures(test_filename),
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=True)

    data = iter(train_loader).next()
    x, y = data[0].to(device), data[1].to(device)

    net = allconvnet64().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0003, eps=1e-4)
    loss = nn.CrossEntropyLoss()

    ll, mm = len(train_loader), len(test_loader)
    ac_, er_ = [], []

    for e in range(epochs):
        rloss = 0.0
        if ((e+1) % 100) == 0:
            optimizer.param_groups[-1]['lr'] /= 2
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
                                        num_classes=11, num_samples=100)
            ac_.append(ac)
            er_.append(er)
            print("Epoch: %d, Accuracy: %f, Accuracy Error: %f" % (e, ac, er))

        print("Epoch: %d, Loss: %f" % (e, rloss/ll))
    torch.save(net, "dvs_gestures_trained_model_full_new.pt")
    ac_, er_ = np.array(ac_), np.array(er_)
    np.save("./data/det_error_dvs_1", er_)
    np.save("./data/det_accur_dvs_1", ac_)

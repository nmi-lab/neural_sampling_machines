import os
import torch
import numpy as np
from readerfy import Events
import torch.utils.data as data
from torchvision import transforms


def to_categorical(x, num_classes):
    return np.eye(num_classes, dtype=np.uint8)[x]


def convert_dvs_pixel():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    ts_train = []
    ts_test = []
    polarity_of_interest = 1

    testdirs = os.listdir('Test')
    traindirs = os.listdir('Train')
    print('Loading Training Data...')
    for i in range(10):
        samples = os.listdir('Train/'+traindirs[i])
        for j in range(len(samples)):
            tmp = []
            filename = 'Train/'+traindirs[i]+'/'+samples[j]
            # print(filename)
            f = open(filename, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()
            raw_data = np.uint32(raw_data)

            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7  # bit 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (
                raw_data[3::5] << 8) | (raw_data[4::5])

            # Process time stamp overflow events
            time_increment = 2 ** 13
            overflow_indices = np.where(all_y == 240)[0]
            for overflow_index in overflow_indices:
                all_ts[overflow_index:] += time_increment

            # Everything else is a proper td spike
            td_indices = np.where(all_y != 240)[0]

            td = Events(td_indices.size, 34, 34)
            td.data.x = all_x[td_indices]
            td.width = td.data.x.max() + 1
            td.data.y = all_y[td_indices]
            td.height = td.data.y.max() + 1
            td.data.ts = all_ts[td_indices]
            td.data.p = all_p[td_indices]
            polid = td.data.p == polarity_of_interest
            tmp.append(td.data.x[polid])
            tmp.append(td.data.y[polid])
            tmp = np.array(tmp)
            tmp = np.concatenate([tmp[0], tmp[1]])
            x_train.append(tmp)
            y_train.append(int(traindirs[i]))
            ts_train.append(td.data.ts[polid])

    print('Loading Test Data...')
    for i in range(10):
        samples = os.listdir('Test/'+testdirs[i])
        for j in range(len(samples)):
            tmp = []
            filename = 'Test/'+testdirs[i]+'/'+samples[j]
            f = open(filename, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()
            raw_data = np.uint32(raw_data)

            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7  # bit 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (
                raw_data[3::5] << 8) | (raw_data[4::5])

            # Process time stamp overflow events
            time_increment = 2 ** 13
            overflow_indices = np.where(all_y == 240)[0]
            for overflow_index in overflow_indices:
                all_ts[overflow_index:] += time_increment

            # Everything else is a proper td spike
            td_indices = np.where(all_y != 240)[0]

            td = Events(td_indices.size, 34, 34)
            td.data.x = all_x[td_indices]
            td.width = td.data.x.max() + 1
            td.data.y = all_y[td_indices]
            td.height = td.data.y.max() + 1
            td.data.ts = all_ts[td_indices]
            td.data.p = all_p[td_indices]
            polid = td.data.p == polarity_of_interest
            print(polid)
            tmp.append(td.data.x[polid])
            tmp.append(td.data.y[polid])
            tmp = np.array(tmp)
            tmp = np.concatenate([tmp[0], tmp[1]])
            x_test.append(tmp)
            y_test.append(int(testdirs[i]))
            ts_test.append(td.data.ts[polid])

    x_train = np.array(x_train)
    np.save('nmnist_train_data.npy', x_train)
    x_test = np.array(x_test)
    np.save('nmnist_test_data.npy', x_test)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train, 10)
    np.save('nmnist_train_labels.npy', y_train)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test, 10)
    np.save('nmnist_test_labels.npy', y_test)
    ts_train = np.array(ts_train)
    np.save('nmnist_train_ts.npy', ts_train)
    ts_test = np.array(ts_test)
    np.save('nmnist_test_ts.npy', ts_test)
    return x_train, y_train, x_test, y_test, ts_train, ts_test


def import_nmnist():
    x_train = np.load('/share/data/nmnist/nmnist_train_data.npy')
    y_train = np.load('/share/data/nmnist/nmnist_train_labels.npy')
    x_test = np.load('/share/data/nmnist/nmnist_test_data.npy')
    y_test = np.load('/share/data/nmnist/nmnist_test_labels.npy')
    ts_train = np.load('/share/data/nmnist/nmnist_train_ts.npy')
    ts_test = np.load('/share/data/nmnist/nmnist_test_ts.npy')
    return x_train, y_train, x_test, y_test, ts_train, ts_test


def threshing():
    x_train, y_train, x_test, y_test, ts_train, ts_test = import_nmnist()

    train_neo = np.zeros([len(x_train), 34, 34, 10])
    test_neo = np.zeros([len(x_test), 34, 34, 10])
    marker = np.arange(10)+1
    for i in range(len(x_train)):
        c1, c2 = np.split(x_train[i], 2)
        binlength = np.max(ts_train[i])/10
        for j in range(10):
            bininds = np.where((ts_train[i] > (binlength*(marker[j]-1))) &
                               (ts_train[i] < (binlength*marker[j])))
            train_neo[i, c1[bininds[0]], c2[bininds[0]], j] = 1

    for i in range(len(x_test)):
        c1, c2 = np.split(x_test[i], 2)
        binlength = np.max(ts_test[i])/10
        for j in range(10):
            bininds = np.where((ts_test[i] > (binlength*(marker[j]-1))) &
                               (ts_test[i] < (binlength*marker[j])))
            test_neo[i, c1[bininds[0]], c2[bininds[0]], j] = 1
    return(train_neo, y_train), (test_neo, y_test)


class NMNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if train is True and os.path.exists("/share/data/nmnist/x_train.npy"):
            print("Loading the data!")
            img_train = np.load("/share/data/nmnist/x_train.npy")
            target_train = np.load("/share/data/nmnist/y_train.npy")
        elif train is False and os.path.exists("/share/data/nmnist/x_test.npy"):
            print("Loading the data!")
            img_test = np.load("/share/data/nmnist/x_test.npy")
            target_test = np.load("/share/data/nmnist/y_test.npy")
        else:
            print("Creating the data!")
            (img_train, target_train), (img_test, target_test) = threshing()

        if train is True:
            self.data, self.targets = img_train, target_train
        else:
            self.data, self.targets = img_test, target_test

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = img.reshape(34, 34, 10)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    batch_size = 32
    kwargs = {'num_workers': 1, 'pin_memory': True}
    dataloader = torch.utils.data.DataLoader(NMNIST('../data',
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()])),
                                             batch_size=batch_size,
                                             shuffle=True)

    data = iter(dataloader)
    print(data.next()[0].shape)

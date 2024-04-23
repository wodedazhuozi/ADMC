import os

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class get_label_data(Dataset):
    def __init__(self, label, data, view):
        self.view = view
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx])], self.label[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx])],\
                self.label[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx]),
                    torch.from_numpy(self.data[3][idx])], self.label[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx]),
                    torch.from_numpy(self.data[3][idx]), torch.from_numpy(self.data[4][idx])], self.label[idx], torch.from_numpy(np.array(idx)).long()

class get_unlabel_data(Dataset):
    def __init__(self, data, view):
        self.view = view
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx])], torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx])],\
                torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx])],\
                torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]), torch.from_numpy(self.data[2][idx]),
                    torch.from_numpy(self.data[3][idx]), torch.from_numpy(self.data[4][idx])], torch.from_numpy(np.array(idx)).long()

class RGBD(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'rgbd_mtv.mat')
        data1 = data['X'][0,0].astype(np.float32)
        data2 = data['X'][0,1].astype(np.float32)
        data3 = data['X'][0,2].astype(np.float32)
        labels = data['gt'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class MNIST_10K(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'MNIST-10K.mat')
        data1 = data['X'][0,0].astype(np.float32)
        data2 = data['X'][1,0].astype(np.float32)
        data3 = data['X'][2,0].astype(np.float32)
        labels = data['y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()


class Reuters1200(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Reuters-1200.mat')
        data1 = data['X'][0,0].astype(np.float32)
        data2 = data['X'][1,0].astype(np.float32)
        data3 = data['X'][2,0].astype(np.float32)
        data4 = data['X'][3,0].astype(np.float32)
        data5 = data['X'][4,0].astype(np.float32)
        labels = data['y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx]), torch.from_numpy(
           self.x4[idx]),torch.from_numpy(self.x5[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class reuters(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'reuters.mat')
        data1 = data['X'][0,0].A.astype(np.float32)
        data2 = data['X'][0,1].A.astype(np.float32)
        data3 = data['X'][0,2].A.astype(np.float32)
        data4 = data['X'][0,3].A.astype(np.float32)
        data5 = data['X'][0,4].A.astype(np.float32)
        labels = data['Y'].transpose().flatten()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx]), torch.from_numpy(
           self.x4[idx]),torch.from_numpy(self.x5[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class Scene15(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'scene-15.mat')
        data1 = data['X'][0,0].astype(np.float32)
        data2 = data['X'][0,1].astype(np.float32)
        data3 = data['X'][0,2].astype(np.float32)
        labels = data['Y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class MSRC_v1(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'MSRC-v1.mat')
        data1 = data['X'][0,0].astype(np.float32)
        data2 = data['X'][0,1].astype(np.float32)
        data3 = data['X'][0,2].astype(np.float32)
        data4 = data['X'][0,3].astype(np.float32)
        labels = data['Y'].transpose().flatten()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(self.x3[idx]), torch.from_numpy(
           self.x4[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'BDGP.mat')
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose().flatten()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class Cora(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Cora.mat')
        data1 = scipy.io.loadmat(path+'Cora.mat')['coracites'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Cora.mat')['coracontent'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Cora.mat')['corainbound'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Cora.mat')['coraoutbound'].T.astype(np.float32)
        labels = scipy.io.loadmat(path+'Cora.mat')['y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(
           self.x4[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class Hdigit(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Hdigit.mat')['data'][0,0].T.astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Hdigit.mat')['data'][0, 1].T.astype(np.float32)
        labels = scipy.io.loadmat(path+'Hdigit.mat')['truelabel'][0,0].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class BBCSport(Dataset):
    def __init__(self, path):
        data0 = scipy.io.loadmat(path+'BBCSport.mat')
        data1 = scipy.io.loadmat(path+'BBCSport.mat')['X'][0, 0].A.T.astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BBCSport.mat')['X'][0, 1].A.T.astype(np.float32)
        labels = scipy.io.loadmat(path+'BBCSport.mat')['gt'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class Animals(Dataset):
    def __init__(self, path):
        data0 = scipy.io.loadmat(path+'Animals.mat')
        data1 = data0['X'][0, 0].astype(np.float32)
        data2 = data0['X'][0, 1].astype(np.float32)
        labels = data0['Y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()


class NoisyMNIST(Dataset):
    def __init__(self, path):
        data0 = scipy.io.loadmat(path+'NoisyMNIST.mat')
        data1 = data0['X'][0, 0].astype(np.float32)
        data2 = data0['X'][0, 1].astype(np.float32)
        labels = data0['Y'].transpose().flatten() - 1
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()
class COIL20(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(os.path.join(path, 'COIL20.mat'))
        X_data = mat['X']
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(X_data[0, 0].astype(np.float32))
        data2 = scaler.fit_transform(X_data[0, 1].astype(np.float32))
        data3 = scaler.fit_transform(X_data[0, 2].astype(np.float32))
        labels = np.array(np.squeeze(mat['Y'])).astype(np.int32) - 1
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(self.x3[idx])], self.y[idx], torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'/CCV/STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'/CCV/SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'/CCV/MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'/CCV/label.npy').flatten()

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], self.labels[idx], torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.tmp = scipy.io.loadmat(path + 'MNIST_USPS.mat')
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose().flatten()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], self.labels[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], self.labels[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view3[idx]), torch.from_numpy(self.view4[idx])], self.labels[idx], torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view5[idx])], self.labels[idx], torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-10K":
        dataset = MNIST_10K('./data/')
        dims = [30, 9, 30]
        view = 3
        class_num = 10
        data_size = 10000
    elif dataset == "Reuters1200":
        dataset = Reuters1200('./data/')
        dims = [2000, 2000, 2000, 2000, 2000]
        view = 5
        class_num = 6
        data_size = 1200
    elif dataset == "reuters":
        dataset = reuters('./data/')
        dims = [21531, 24892, 34251, 15506, 11547]
        view = 5
        class_num = 6
        data_size = 18758
    elif dataset == "RGBD":
        dataset = RGBD('./data/')
        dims = [20, 59, 40]
        view = 3
        class_num = 15
        data_size = 4485
    elif dataset == "Scene15":
        dataset = Scene15('./data/')
        dims = [20, 59, 40]
        view = 3
        class_num = 15
        data_size = 4485
    elif dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 3304, 6750]
        view = 3
        class_num = 20
        data_size = 1440
    elif dataset == "MSRC_v1":
        dataset = MSRC_v1('./data/')
        dims = [24, 512, 256, 254]
        view = 4
        class_num = 7
        data_size = 210
    elif dataset == "Animals":
        dataset = Animals('./data/')
        dims = [4096, 4096]
        view = 2
        class_num = 50
        data_size = 10158
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 30000
    elif dataset == "Cora":
        dataset = Cora('./data/')
        dims = [2708, 1433, 2706, 2706]
        view = 4
        class_num = 7
        data_size = 2708
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        class_num = 10
        data_size = 10000
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "BBCSport":
        dataset = BBCSport('data/')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 1984]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 1984, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 1984, 512, 928]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError

    return dataset, dims, view, data_size, class_num
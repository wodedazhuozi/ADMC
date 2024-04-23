import argparse
import configparser

import numpy as np
import torch
from dataloader import load_data, get_unlabel_data, get_label_data
from network import Network
from sklearn.cluster import KMeans
from utils import distance_euclidean_scipy, unlabel_loss, evaluate, Loss
import torch.nn as nn
import torch.optim as optim
import os
import warnings
import time

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# BBCSport
# Hdigit
# COIL20
# Cora
# MSRC_v1
# Scene15
# Reuters1200
# reuters
# RGBD
# MNIST-10K
# Animals
# NoisyMNIST
Dataname = 'Cora'

config = configparser.ConfigParser()
config_path = './config.ini'
config.read(config_path)

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.6)
# parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=config.getfloat(Dataname, 'unlabeled_lr'))
parser.add_argument("--labeled_lr", default=config.getfloat(Dataname, 'labeled_lr'))
parser.add_argument("--weight_decay", default=3e-4)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--labeled_epochs", default=config.getint(Dataname, 'labeled_epochs'))
parser.add_argument("--unlabeled_epochs", default=config.getint(Dataname, 'unlabeled_epochs'))
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--annotation_num_epoch", default=config.getint(Dataname, 'annotation_num_epoch'))
parser.add_argument("--threshold", default=0.95)
parser.add_argument("--smoothing", default=0.6)
parser.add_argument("--EPOCH", default=10)
parser.add_argument("--alphabet", default=config.getfloat(Dataname, 'alphabet'))
parser.add_argument("--beta", default=config.getfloat(Dataname, 'beta'))
parser.add_argument("--gama", default=config.getfloat(Dataname, 'gama'))
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["OMP_NUM_THREADS"] = '' + str(args.annotation_num_epoch)



dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)
optimizer = optim.AdamW(model.parameters(),
                           lr=args.learning_rate,
                           betas=(0.9, 0.998),
                           eps=1e-08,
                           weight_decay=1e-4,
                           amsgrad=False)
optimizer_labeled = optim.AdamW(model.parameters(),
                           lr=args.labeled_lr,
                           betas=(0.9, 0.998),
                           eps=1e-08,
                           weight_decay=1e-4512,
                           amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.labeled_epochs * args.EPOCH * 2)
scheduler_label = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_labeled, args.labeled_epochs * args.EPOCH * 2)

def pretrain(Epoch, path):
    for epoch in range(Epoch):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            _, _, xrs, _, _, _, _, _ = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        if epoch % 100 == 0:
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

        torch.save(model.state_dict(), path + '/network.pt')


def get_data():
    Data = []
    real_label = []

    for v in range(view):
        Data.append([])

    for batch_idx, (xs, label, index) in enumerate(data_loader):  #
        for v in range(view):
            Data[v].extend(xs[v].cpu().detach().numpy())
        real_label.extend(label)

    for v in range(view):
        Data[v] = np.array(Data[v])

    real_label = np.array(real_label)
    return Data, real_label

def splite_data(dataset, reallabel):
    dataset = get_label_data(reallabel, dataset, view)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    Data = []
    dataset_index = []
    pre_label = []
    real_label = []
    fusion_soft_labels = []
    fusion_features = []
    soft_labels = []
    annotation_data = []
    pseu_data = []
    uncertain_data = []
    label_data = []
    soft_prelabel = []
    for v in range(view):
        Data.append([])
        pre_label.append([])
        soft_labels.append([])
        annotation_data.append([])
        pseu_data.append([])
        uncertain_data.append([])
        label_data.append([])
        soft_prelabel.append([])

    for batch_idx, (xs, label, index) in enumerate(data_loader):  #
        for v in range(view):
            Data[v].extend(xs[v].cpu().detach().numpy())
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, soft_label, _, _, fusion_soft_label, fusion_feature, _, _ = model(xs)
        fusion_features.extend(fusion_feature.cpu().detach().numpy())
        fusion_soft_labels.extend(fusion_soft_label.cpu().detach().numpy())
        real_label.extend(label)
        dataset_index.extend(index)

        for v in range(view):
            soft_labels[v].extend(soft_label[v].cpu().detach().numpy())
            predict_cla = np.argmax(soft_label[v].cpu().detach().numpy(), axis=1)
            pre_label[v].extend(predict_cla)
    for v in range(view):
        pre_label[v] = np.array(pre_label[v])
        soft_labels[v] = np.array(soft_labels[v])
        soft_prelabel[v] = np.amax(soft_labels[v], axis=1)
        Data[v] = np.array(Data[v])
    dataset_index = np.array(dataset_index)
    fusion_features = np.array(fusion_features)
    fusion_soft_labels = np.array(fusion_soft_labels)
    real_label = np.array(real_label)

    # find annotation data
    kmeans = KMeans(n_clusters=args.annotation_num_epoch, n_init=50, random_state=0).fit(fusion_features)
    centers = kmeans.cluster_centers_
    distance = distance_euclidean_scipy(centers, fusion_features)
    annotation_index = np.argmin(distance, axis=1)

    for v in range(view):
        annotation_data[v] = Data[v][annotation_index]
    annotation_label = real_label[annotation_index]
    real_label = np.delete(real_label, annotation_index, 0)

    # find pseudo label data
    for v in range(view):
        Data[v] = np.delete(Data[v], annotation_index, 0)
        soft_labels[v] = np.delete(soft_labels[v], annotation_index, 0)
        soft_prelabel[v] = np.delete(soft_prelabel[v], annotation_index, 0)
    fusion_soft_labels = np.delete(fusion_soft_labels, annotation_index, 0)
    soft_prelabel.append(np.amax(fusion_soft_labels, axis=1))
    condition = np.all(np.array(soft_prelabel) >= np.full((len(soft_prelabel[0])), fill_value=args.threshold), axis=0)

    pseu_d_ind = np.where(condition)
    # pseu_d_ind = np.where(
    #     np.amax(fusion_soft_labels, axis=1) >= np.full((len(soft_prelabel[0])), fill_value=args.threshold))
    for v in range(view):
        pseu_data[v] = Data[v][pseu_d_ind]
    pseu_data_label = pre_label[0][pseu_d_ind]

    # find uncertain data
    real_label = np.delete(real_label, pseu_d_ind, 0)
    for v in range(view):
        uncertain_data[v] = np.delete(Data[v], pseu_d_ind, 0)

    for v in range(view):
        label_data[v] = np.concatenate((annotation_data[v], pseu_data[v]), axis=0)
    labeled_label = np.concatenate((annotation_label, pseu_data_label), axis=0)
    # np.delete(fusion_features, pseu_d_ind, 0)
    other = np.delete(fusion_features, pseu_d_ind, 0)
    other = np.delete(other, annotation_index, 0)

    return label_data, labeled_label, uncertain_data, real_label, other


def train_label_data(data, label, view, smoothing, Epoch):
    model.train()
    # criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    mes = torch.nn.MSELoss()
    dataset = get_label_data(label, data, view)
    for epoch in range(Epoch):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        tot_loss = 0
        for batch_idx, (xs, label, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            hs, qs, xrs, zs, _, _, logits, f_logits = model(xs)
            criteria = nn.CrossEntropyLoss(label_smoothing=smoothing)
            logits.append(f_logits)
            loss1 = 0
            for v in range(view + 1):
                loss1 += criteria(logits[v], label.to(device))
            tot_loss += loss1.item()

            loss_list = []
            for v in range(view):
                # for w in range(v + 1, view):
                #     loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = args.alphabet * sum(loss_list)+loss1

            loss.requires_grad_(True)

            # 更新网络参数
            optimizer_labeled.zero_grad()  # 将网络中所有的参数的导数都清0
            loss.backward()  # 计算梯度
            optimizer_labeled.step()  # 更新参数
        scheduler_label.step()
        if epoch % 100 == 0:
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def train_unlabel_data(data, view, Epoch, feature):
    model.train()
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    kmeans = KMeans(n_clusters=class_num, n_init=50, random_state=0).fit(feature)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(device)
    labels = kmeans.labels_
    dataset = get_label_data(labels, data, view)
    for epoch in range(Epoch):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        for batch_idx, (xs, label, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            hs, qs, xrs, zs, _, f_feature, _, _ = model(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(args.beta * mes(xs[v], xrs[v]))
            for row1, row2 in zip(f_feature, label):
                loss_list.append(args.gama * torch.norm(row1 - centers[row2.item()]) * torch.norm(row1 - centers[row2.item()]))
            loss = sum(loss_list)

            optimizer.zero_grad()  # 将网络中所有的参数的导数都清0
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()
        # if epoch % 100 == 0:
        #     print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def test(data, label):
    model.eval()
    dataset = get_label_data(label, data, view)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    real_label = []
    pre_label = []
    for batch_idx, (xs, label, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, _, _, fusion_soft_label, _, _, _ = model(xs)
        real_label.extend(label)
        pre_label.extend(fusion_soft_label.cpu().detach().numpy())
    pre_label = np.array(pre_label)
    real_label = np.array(real_label)
    pre_label = np.argmax(pre_label, axis=1)
    evaluate(real_label, pre_label)


if __name__ == '__main__':
    start = time.time()
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    path = "./" + Dataname + "_result"
    if not os.path.exists(path + "/network.pt"):
        os.mkdir(path)
        pretrain(args.mse_epochs, path)
    else:
        model.load_state_dict(torch.load(path + "/network.pt"))
    unlabeled_data, uncertain_data_label = get_data()
    data = unlabeled_data
    label = uncertain_data_label
    test(data, label)
    labeled_data, labeled_label, unlabeled_data, uncertain_data_label, fusion_feature = splite_data(unlabeled_data, uncertain_data_label)
    for epoch in range(5):
        # for v in range(10):
        print("labeled data number:", len(labeled_data[0]))

        # print("After train labeled: ")
        # train_label_data(labeled_data, labeled_label, view, args.smoothing, args.labeled_epochs)
        # test(data, label)

        print("After train unlabeled: ")
        train_unlabel_data(unlabeled_data, view, args.unlabeled_epochs, fusion_feature)
        # args.unlabeled_epochs = args.unlabeled_epochs + 10
        test(data, label)

        print("After train labeled: ")
        train_label_data(labeled_data, labeled_label, view, args.smoothing, args.labeled_epochs)
        test(data, label)
        new_labeled_data, new_labeled_label, unlabeled_data, uncertain_data_label, fusion_feature = splite_data(unlabeled_data,
                                                                                        uncertain_data_label)
        for v in range(view):
            labeled_data[v] = np.concatenate((labeled_data[v], new_labeled_data[v]), axis=0)
        labeled_label = np.concatenate((labeled_label, new_labeled_label), axis=0)
    print("all time:", time.time() - start)
    torch.save(model.state_dict(), path + '/final_network.pt')

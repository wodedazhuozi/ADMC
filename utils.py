import numpy as np
import torch
from scipy import spatial
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import os
os.environ["OMP_NUM_THREADS"] = '1'

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * len(h_i)
        # N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, len(h_i))
        sim_j_i = torch.diag(sim, -len(h_i))
        # sim_i_j = torch.diag(sim, self.batch_size)
        # sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((torch.unsqueeze(sim_i_j, 1).transpose(0, 1), torch.unsqueeze(sim_j_i, 1).transpose(0, 1)), dim=0)
        positive_samples = positive_samples.reshape(N, 1)
        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def distance_euclidean_scipy(vec1, vec2, distance="euclidean"):
    return spatial.distance.cdist(vec1, vec2, distance)

def unlabel_loss(feature):
    length = len(feature)
    features = feature[0]
    for v in range(length - 1):
        features = torch.stack((features, feature[v + 1]), 1)
    center = torch.unsqueeze(torch.mean(features[0], 0), 1).transpose(0, 1)
    for v in range(len(features) - 1):
        ax = torch.unsqueeze(torch.mean(features[v + 1], 0), 1).transpose(0, 1)
        center = torch.cat((center, ax), 0)
    distance = torch.sqrt(torch.sum((feature[0] - center) ** 2, dim=1))
    for v in range(len(feature) - 1):
        distance = distance + torch.sqrt(torch.sum((feature[v + 1] - center) ** 2, dim=1))
    return distance.sum()

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):

    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    print('ACC:{:.4f}'.format(acc))
    print('NMI:{:.4f}'.format(nmi))
    print('ARI:{:.4f}'.format(ari))
    print('PUR:{:.4f}'.format(pur))

    return acc, nmi, ari, pur
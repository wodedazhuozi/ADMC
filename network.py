import torch.nn as nn
from torch.nn.functional import normalize
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Weight_layer(nn.Module):
    def __init__(self, input_dim):
        super(Weight_layer, self).__init__()
        self.weight = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.weight(x)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # 计算query、key和value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算注意力权重
        weights = torch.matmul(q, k.transpose(-2, -1))
        weights = weights / torch.sqrt(torch.tensor(self.hidden_dim).float())
        weights = torch.softmax(weights, dim=-1)

        # 加权求和得到输出
        output = torch.matmul(weights, v)

        return output

class Fusion(nn.Module):
    def __init__(self, view, input_dim, feature_dim):
        super(Fusion, self).__init__()
        self.view = view
        self.FC = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, xs):
        x = torch.full([xs[0].length,],0.)
        for v in range(self.view):
            x = x + xs[v]
        return self.FC(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.weight = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.weight.append(Weight_layer(feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.weight = nn.ModuleList(self.weight)
        # self.fusion = Fusion(view, feature_dim, class_num)

        # self.attention = SelfAttention(feature_dim,feature_dim)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num)
        )
        self.softmax = nn.Softmax(dim=1)
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        logits = []
        ws = []
        for v in range(self.view):
            x = xs[v]
            z  = self.encoders[v](x)
            weight = self.weight[v](z)
            # z_attention = self.attention(z)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            logit = self.label_contrastive_module(z)
            q = self.softmax(logit)
            xr = self.decoders[v](z)
            logits.append(logit)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
            ws.append(weight)

        weights = torch.cat(ws,dim=1)
        weights = self.softmax(weights)
        fx = torch.full(zs[0].shape, 0).to(device)
        for v in range(self.view):
            fx = fx + torch.mul(zs[v], weights[:, v].unsqueeze(1))
        f_logits = self.label_contrastive_module(fx)

        return hs, qs, xrs, zs, self.softmax(f_logits), fx, logits, f_logits
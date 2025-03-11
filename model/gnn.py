import torch
import torch.nn as nn
import torch.nn.functional as F

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, x, node_anchor_adj):
        a_in = torch.matmul(node_anchor_adj, x)
        a_out = torch.matmul(node_anchor_adj.transpose(1, 2), x)
        a = torch.cat((a_in, a_out, x), dim=2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * x + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, args, num_classes):
        super(GGNN, self).__init__()

        self.state_dim = args.rnn_units
        self.n_edge_types = args.num_edge_types
        self.n_steps = args.n_steps
        self.nclass = num_classes

        self.propagator = Propogator(self.state_dim)


        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, node_anchor_adj):
        for i_step in range(self.n_steps):
            x = self.propagator(x, node_anchor_adj)

        output = x

        return output

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())

class GCN(torch.nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(GCN, self).__init__()
        self.num_classes = num_classes
        self.num_nodes = args.num_nodes
        self.rnn_units = args.rnn_units
        self.seq_len = args.max_seq_len
        self.input_dim = args.input_dim
        self._device = device

        # 更新卷积层大小以减少参数量
        self.conv1 = GCNLayer(self.seq_len * self.input_dim, 512)  # 可尝试 256 以进一步减少
        self.conv2 = GCNLayer(512, self.rnn_units)  # 将rnn_units减半

        # 全连接层
        self.fc = torch.nn.Linear(self.rnn_units, self.num_classes)

    def forward(self, x, adj):
        batch_size = x.shape[0]

        # 对每个节点的特征展平时间和输入维度
        x = x.view(batch_size, self.num_nodes, -1)  # 变成 [48, 18, 1200]

        # 逐节点处理
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.conv2(x, adj)
        x = F.relu(x)

        # 汇总所有节点的特征
        x = torch.mean(x, dim=1)  # [48, rnn_units // 2]
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

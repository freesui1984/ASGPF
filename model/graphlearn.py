import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_cuda
from constants import VERY_SMALL_NUMBER, INF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence





def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):

    def __init__(self, args, input_size, hidden_size, epsilon=None, num_pers=16, device=None):
        super(GraphLearner, self).__init__()
        self.args = args
        self.device = device
        self.epsilon = epsilon
        self.dropout = args.dropout
        self.num_pers = num_pers

        self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
        self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
        self.att = EEGGraphAttentionLayer(input_size, hidden_size, self.device)
        self.leakyrelu = nn.LeakyReLU(0.2)



    def forward(self, context, adj, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        attention_head = []
        attention = []
        context = context.reshape(adj.size(0), adj.size(1), -1)

        for _ in range(self.num_pers):
            for i in range(context.size(0)):
                h = context[i]
                ad = adj[i]
                attention_ = self.att(h, ad)
                attention_head.append(attention_)
            attention_head = torch.stack(attention_head, 0)
            attention.append(attention_head)
            attention_head = []

        attention = torch.mean(torch.stack(attention, 0), 0)
        markoff_value = -INF


        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        attention = F.softplus(attention)

        return attention


    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists


def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1)
    adj = to_cuda(torch.zeros_like(attention).scatter_(-1, knn_ind, 1), device)
    return adj

class EEGGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, device=None, dropout=0.5, alpha=0.2, concat=True):
        super(EEGGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device

        self.w = nn.Parameter(torch.empty(size=(in_features, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).to(self.device)

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)

        Wh = torch.matmul(h, self.w)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention


    def _prepare_attentional_mechanism_input(self, Wh):
        # number of nodes
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def dropout(x, drop_prob, shared_axes=[], training=False):
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, \
                 bidirectional=False, num_layers=1, rnn_type='lstm', rnn_dropout=None, device=None):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using {}-layer bidirectional {} encoder ]'.format(num_layers, rnn_type))
        else:
            print('[ Using {}-layer {} encoder ]'.format(num_layers, rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        self.to(device)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size),
                         self.device)
            packed_h, (packed_h_t, packed_c_t) = self.model(x.float(), (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
            if self.rnn_type == 'lstm':
                packed_c_t = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]
            if self.rnn_type == 'lstm':
                packed_c_t = packed_c_t[-1]

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]
        restore_hh = dropout(restore_hh, self.rnn_dropout, shared_axes=[-2], training=self.training)
        restore_hh = restore_hh.transpose(0, 1)  # [max_length, batch_size, emb_dim]

        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_packed_h_t = dropout(restore_packed_h_t, self.rnn_dropout, training=self.training)
        restore_packed_h_t = restore_packed_h_t.unsqueeze(0)  # [1, batch_size, emb_dim]

        if self.rnn_type == 'lstm':
            restore_packed_c_t = packed_c_t[inverse_indx]
            restore_packed_c_t = dropout(restore_packed_c_t, self.rnn_dropout, training=self.training)
            restore_packed_c_t = restore_packed_c_t.unsqueeze(0)  # [1, batch_size, emb_dim]
            rnn_state_t = (restore_packed_h_t, restore_packed_c_t)
        else:
            rnn_state_t = restore_packed_h_t
        return restore_hh, rnn_state_t

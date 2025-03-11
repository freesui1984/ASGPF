import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.graphlearn import GraphLearner, get_binarized_kneighbors_graph, dropout, EncoderRNN
from constants import VERY_SMALL_NUMBER, INF
from utils import to_cuda
import utils



class EEGGraphClf(nn.Module):
    def __init__(self, args, device):
        super(EEGGraphClf, self).__init__()
        self.args = args
        self.name = 'EEGGraphClf'
        self.device = device

        # Shape
        self.embed_dim = args.input_dim * args.max_seq_len
        self.output_dim = args.rnn_units * args.max_seq_len
        self.hidden_size = args.rnn_units * args.max_seq_len
        self.nclass = args.num_classes
        self.num_nodes = args.num_nodes

        # Dropout
        self.dropout = args.dropout
        self.word_dropout = args.dropout
        self.rnn_dropout = args.dropout

        # Graph
        self.graph_learn = args.use_graph_learning
        self.graph_skip_conn = args.graph_skip_conn
        self.graph_include_self = args.graph_include_self

        self.fc = nn.Linear(args.rnn_units, self.nclass, device=self.device, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()


        if self.graph_learn:
            self.graph_learner = GraphLearner(args, self.embed_dim, args.graph_learn_hidden_size,
                                                epsilon=args.graph_learn_epsilon,
                                                num_pers=args.graph_learn_num_pers,
                                                device=self.device)

            self.graph_learner2 = GraphLearner(args, self.hidden_size,
                                                args.graph_learn_hidden_size,
                                                epsilon=args.graph_learn_epsilon,
                                                num_pers=args.graph_learn_num_pers,
                                                device=self.device)

            self.ctx_rnn_encoder = EncoderRNN(self.embed_dim, self.output_dim, bidirectional=True, num_layers=1, rnn_type='lstm'
                                              , rnn_dropout=self.rnn_dropout, device=self.device)



    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, anchor_mask=None, graph_include_self=False, init_adj=None, anchor_features=None):
        if self.graph_learn:
            # raw_adj: attentional score
            raw_adj = graph_learner(node_features, init_adj, node_mask)
            adj = torch.softmax(raw_adj, dim=-1)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
            else:
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

            return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def compute_output(self, node_vec, seq_lengths, node_mask=None):
        batch_size, max_seq_length = node_vec.shape[0], node_vec.shape[1]
        output = node_vec.reshape(batch_size, max_seq_length, -1)

        # extract last relevant output
        last_out = utils.last_relevant_pytorch(
            output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, -1)
        last_out = last_out.to(self.device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits

    def graph_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embedding

    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.args.smoothness_ratio * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape[1:])))

            graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)

            log_degrees = torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)


            total_degrees = torch.matmul(ones_vec.unsqueeze(1), log_degrees).squeeze(-1).squeeze(-1)

            degree_regularization = -self.args.degree_ratio * total_degrees / out_adj.shape[-1]

            graph_loss += degree_regularization
            graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.args.smoothness_ratio * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.args.degree_ratio * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / out_adj.shape[-1]
            graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        return graph_loss

    def batch_SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

    def batch_diff(self, X, Y, Z):
        assert X.shape == Y.shape
        diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2))  # Shape: [batch_size]
        norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
        diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
        return diff_

    def prepare_init_graph(self, context, context_lens, training=True):
        raw_context_vec = dropout(context, self.word_dropout, shared_axes=[-2], training=training)
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[0].transpose(0, 1)
        return raw_context_vec, context_vec

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # 根据类别设置权重
            alpha_factor = torch.tensor(self.alpha, device=input.device)[target]
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


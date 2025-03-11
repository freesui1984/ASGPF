import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(input_dim, num_heads, hidden_dim) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(input_dim, num_heads, hidden_dim) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return tgt


class TransformerModel(nn.Module):
    def __init__(self, args, device=None):
        super(TransformerModel, self).__init__()

        num_nodes = args.num_nodes
        num_layers = args.num_layers
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        dropout = args.dropout

        self.encoder = TransformerEncoder(input_dim=args.input_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = TransformerDecoder(input_dim=args.output_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
        self.projection_layer = nn.Linear(hidden_dim, args.output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None):
        encoder_outputs = self.encoder(encoder_inputs, mask=encoder_mask)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, tgt_mask=decoder_mask, memory_mask=encoder_mask)
        projected = self.projection_layer(self.dropout(decoder_outputs))
        return projected

########## TransformerModel for seizure classification/detection ##########
class TransformerModel_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(TransformerModel_classification, self).__init__()

        num_nodes = args.num_nodes
        num_layers = args.num_rnn_layers
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        dropout = args.dropout
        enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = enc_input_dim
        self.num_classes = num_classes
        self._device = device

        self.encoder = TransformerEncoder(input_dim=enc_input_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(12*19*125, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]
        input_seq = input_seq.view(batch_size, -1, self.input_dim)

        # encoder
        encoder_outputs = self.encoder(input_seq)

        # extract last relevant output
        last_out = encoder_outputs
        # (batch_size, num_nodes, hidden_dim)
        last_out = last_out.view(batch_size, -1)
        last_out = last_out.to(self._device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
########## TransformerModel for seizure classification/detection ##########

########## TransformerModel for next time prediction ##########
class TransformerModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(TransformerModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_layers = args.num_layers
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        dropout = args.dropout

        self.num_nodes = args.num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = args.output_dim
        self._device = device
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = TransformerEncoder(input_dim=args.input_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = TransformerDecoder(input_dim=args.output_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
        self.projection_layer = nn.Linear(hidden_dim, args.output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        encoder_inputs,
        decoder_inputs,
        encoder_mask=None,
        decoder_mask=None,
        batches_seen=None
    ):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            encoder_mask: mask for encoder inputs
            decoder_mask: mask for decoder inputs
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # encoder
        encoder_outputs = self.encoder(encoder_inputs, mask=encoder_mask)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        decoder_outputs = self.decoder(
            decoder_inputs,
            encoder_outputs,
            tgt_mask=decoder_mask,
            memory_mask=encoder_mask)
        projected = self.projection_layer(self.dropout(decoder_outputs))
        outputs = projected.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## TransformerModel for next time prediction ##########
import math
import random
import torch.nn.functional as F
import torch
import torch.nn as nn

from nlper.utils.torch_utils import get_device
from nlper.utils.lang_utils import Token


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=1).to(get_device())
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True).to(get_device())

    def forward(self, sequence, hidden=None):
        embedding_output = self.embedding(sequence)  # max_text_len x batch_size x embedding_size
        encoder_outputs, hidden = self.gru(embedding_output, hidden)
        # hidden: bidirectional x batch_size x hidden_size
        # output: max_text_len x batch_size x bidirectional * hidden_size
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        # output: max_text_len x batch_size x hidden_size
        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size).to(get_device())
        self.v = nn.Parameter(torch.rand(hidden_size)).to(get_device())
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.transpose(0, 1).repeat(1, timestep, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)  # batch_size x t x hidden
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # batch_size x t

    def score(self, hidden, encoder_outputs):
        # batch_size x t x 2*hidden -> batch_size x t x hidden
        energy = torch.tanh(self.attention(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # batch_size x t x 2*hidden -> batch_size x t x hidden
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # batch_size x 1 x hidden
        energy = torch.bmm(v, energy)  # batch_size x 1 x t
        return energy.squeeze(1)  # batch_size x t


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            output_size, embedding_size, padding_idx=1).to(get_device())
        self.dropout = nn.Dropout(dropout, inplace=True).to(get_device())
        self.attention = BahdanauAttention(hidden_size).to(get_device())
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout).to(get_device())
        self.output = nn.Linear(hidden_size * 2, output_size).to(get_device())

    def forward(self, sequence, hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedding_output = self.embedding(sequence).unsqueeze(0)  # 1 x batch_size x n
        embedding_output = self.dropout(embedding_output)
        # Calculate attention weights and apply to encoder outputs
        attention_weights = self.attention(hidden, encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))  # batch_size x 1 x n
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        decoder_input = torch.cat([embedding_output, context], 2)
        decoder_output, hidden = self.gru(decoder_input, hidden)
        decoder_output = decoder_output.squeeze(0)  # (1,B,N) -> (B,N)
        decoder_output = self.output(torch.cat([decoder_output, context.squeeze(0)], 1))
        decoder_output = F.log_softmax(decoder_output, dim=1)
        return decoder_output, hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text, summary, teacher_forcing_ratio=0.5):
        batch_size = text.size(1)
        max_len = summary.size(0)
        vocab_size = self.decoder.output_size

        encoder_output, hidden = self.encoder(text)
        hidden = hidden[:self.decoder.n_layers]
        output = summary.data[0, :]  # sos

        outputs = torch.FloatTensor(max_len, batch_size, vocab_size).fill_(0).to(get_device())
        for t in range(1, max_len):
            output, hidden, attention_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top_first = output.data.max(1)[1]
            output = summary.data[t] if is_teacher else top_first
        return outputs

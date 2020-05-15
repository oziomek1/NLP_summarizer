import math
import random
import torch.nn.functional as F
import torch
import torch.nn as nn

from typing import Any
from typing import Tuple

from nlper.utils.torch_utils import get_device


class EncoderRNN(nn.Module):
    """
    Model encoder class
    Initializes embedding laayer and bidirectional GRU.

    :param input_size: Number of unique words in vocabulary
    :type input_size: int
    :param embedding_size: Size of embedding layer, number of expected features in GRU
    :type embedding_size: int
    :param hidden_size: Number of features in the hidden state of GRU
    :type hidden_size: int
    :param n_layers: Number of recurrent layers in GRU
    :type n_layers: int
    :param dropout: Probability of dropout on GRU layer except from last layer
    :type dropout: float
    """
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, n_layers: int = 1, dropout: float = 0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=1).to(get_device())
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True).to(get_device())

    def forward(self, sequence: torch.Tensor, hidden: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines encoder structure and flow.

        * Pushes sequence through embedding layer
        * Feeds GRU with embedded sequence
        * Merges bidirectional GRU model into single tensor

        :param sequence: Tensor of indices representing text
        :type sequence: torch.Tensor
        :param hidden: Initial hidden state of GRU, default None
        :type hidden: torch.Tensor, optional
        :return:
        """
        embedding_output = self.embedding(sequence)  # max_text_len x batch_size x embedding_size
        encoder_outputs, hidden = self.gru(embedding_output, hidden)
        # hidden: bidirectional x batch_size x hidden_size
        # output: max_text_len x batch_size x bidirectional * hidden_size
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        # output: max_text_len x batch_size x hidden_size
        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention
    Initializes fully connected layer and internal parameter V with uniformly distributed weights.

    :param hidden_size: Number of features attention in fully connected layer
    :type hidden_size: int
    """
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size).to(get_device())
        self.v = nn.Parameter(torch.rand(hidden_size)).to(get_device())
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention weights by applying softmax on attention alignment scores.

        :param hidden: Encoder hidden states
        :type hidden: torch.Tensor
        :param encoder_outputs: Encoder outputs
        :type encoder_outputs: torch.Tensor
        :return: Attention weights
        :rtype: torch.Tensor
        """
        h = hidden.transpose(0, 1).repeat(1, encoder_outputs.size(0), 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)  # batch_size x t x hidden
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # batch_size x t

    def score(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculates alignment scores of attention.

        :param hidden: Encoder hidden states
        :type hidden: torch.Tensor
        :param encoder_outputs: Encoder outputs
        :type encoder_outputs: torch.Tensor
        :return: Attention alignment scores
        :rtype: torch.Tensor
        """
        # batch_size x t x 2*hidden -> batch_size x t x hidden
        energy = torch.tanh(self.attention(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # batch_size x t x 2*hidden -> batch_size x t x hidden
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # batch_size x 1 x hidden
        energy = torch.bmm(v, energy)  # batch_size x 1 x t
        return energy.squeeze(1)  # batch_size x t


class DecoderRNN(nn.Module):
    """
    Model decoder class
    Initializes embedding layer, dropout layer, Bahdanau attention module, single directional GRU and linear classifier.

    :param embedding_size: Size of embedding layer, number of expected features in GRU
    :type embedding_size: int
    :param hidden_size: Number of features in the hidden state of GRU and in fully connected layer of attention
    :type hidden_size: int
    :param output_size: Number of unique words in vocabulary
    :type output_size: int
    :param n_layers: Number of recurrent layers in GRU
    :type n_layers: int
    :param dropout: Probability of dropout on GRU layer except from last layer
    :type dropout: float
    """
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            output_size: int,
            n_layers: int = 1,
            dropout: float = 0.1,
    ):
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
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers).to(get_device())
        self.classifier = nn.Linear(hidden_size * 2, output_size).to(get_device())

    def forward(self, sequence: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines decoder structure and flow.

        * Pushes sequence through embedding layer
        * Applies dropout
        * Calls attention layer to obtain attention weights
        * Calculates context vector of attention
        * Concatenates context vector with previous decoder output
        * Feeds GRU with concatenation result
        * Generate final output by applying softmax

        :param sequence: StartOfSentence token or previous decoder output
        :type sequence: torch.Tensor
        :param hidden: Hidden state
        :type hidden: torch.Tensor
        :param encoder_outputs: Encoder output
        :type encoder_outputs: torch.Tensor
        :return: Decoder output, decoder hidden state and attention weights
        :rtype: tuple
        """
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
        decoder_output = self.classifier(torch.cat([decoder_output, context.squeeze(0)], 1))
        return decoder_output, hidden, attention_weights


class Seq2Seq(nn.Module):
    """
    Sequence to Sequence model, built using encoder and decoder.

    :param encoder: Encoder model
    :type encoder: nn.Module
    :param decoder: Decoder model with Bahdanau attention
    :type decoder: nn.Module
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text: torch.Tensor, summary: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Defines Seq2Seq structure and flow.
        Teacher forcing ratio specifies probability of altering the decoder output with the target summary token
        for the next word generation. Used to accelerate model learning time.

        * Feeds encoder with input indices
        * Initializes decoder hidden state as encoder hidden state
        * Initializes decoder output with Start of Sequence <sos> token
        * Initializes summary output vector
        * Until the maximum summary length is reached:
            * Feeds decoder with decoder output, hidden state and encoder output
            * Updates decoder output and hidden state
            * Updates summary output vector with decoder output token
            * With teacher_forcing_ratio probability alters decoder output

        :param text: Indices of input text
        :type text: torch.Tensor
        :param summary: Indices of target / reference summary
        :type summary: torch.Tensor
        :param teacher_forcing_ratio:
        :type teacher_forcing_ratio: float
        :return: Output sequence / summary
        :rtype: torch.Tensor
        """
        batch_size = text.size(1)
        max_len = summary.size(0)
        vocab_size = self.decoder.output_size

        encoder_output, hidden = self.encoder(text)
        hidden = hidden[:self.decoder.n_layers]
        output = summary.data[0, :]

        outputs = torch.FloatTensor(max_len, batch_size, vocab_size).fill_(0).to(get_device())
        for t in range(1, max_len):
            output, hidden, attention_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top_first = output.data.max(1)[1]
            output = summary.data[t] if is_teacher else top_first
        return outputs

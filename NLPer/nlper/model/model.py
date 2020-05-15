import logging
import numpy as np
import torch
import torch.nn as nn

from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from nlper.utils.lang_utils import Token
from nlper.utils.torch_utils import get_device
from nlper.utils.torch_utils import AVAILABLE_GPU
from nlper.model.architecture import EncoderRNN
from nlper.model.architecture import DecoderRNN
from nlper.model.architecture import Seq2Seq

from nlper.utils.train_utils import calculate_rouge
from nlper.utils.train_utils import draw_attention_matrix


class Model:
    """
    Utils of Seq2Seq model.

    Executes:
    * model training
    * text prediction (summarization)
    * model evaluation
    * saving and loading of a model

    :param config: Config dictionary
    :type config: dict
    :param vocab_config: Vocabulary config for model
    :type vocab_config: VocabConfig
    """
    def __init__(self, config: Dict[str, Any], vocab_config: Any):
        self.logger = logging.getLogger(Model.__name__)
        self.config = config
        self.vocab_config = vocab_config
        self.encoder = None
        self.decoder = None
        self.seq2seq = None
        self.criterion = None
        self.scheduler = None
        self.optimizer = None
        self.create_model()

    def create_model(self) -> None:
        """
        Initializes full Seq2Seq model with encoder and decoder as specified in config file.
        """
        self.encoder = EncoderRNN(
            input_size=self.config['text_size'],
            embedding_size=self.config['embed_size'],
            hidden_size=self.config['hidden_size'],
            n_layers=2,
            dropout=0.5,
        )
        self.decoder = DecoderRNN(
            embedding_size=self.config['embed_size'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['text_size'],
            n_layers=1,
            dropout=0.5,
        )
        self.seq2seq = Seq2Seq(encoder=self.encoder, decoder=self.decoder).to(get_device())

    def create_optimizers_and_loss(self) -> None:
        """
        Initializes Adam optimizer for Seq2Seq model and learning rate scheduler as specified in config file.
        Initialized CrossEntropyLoss with ignoring padding token <pad> from sequence.
        """
        self.optimizer = optim.Adam(self.seq2seq.parameters(), lr=self.config['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['scheduler_step_size'],
            gamma=self.config['scheduler_gamma'],
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_config.stoi[Token.Padding.value]).to(get_device())

    def evaluate(self, valid_iterator: Any) -> List[torch.Tensor]:
        """
        Evaluates the trained Seq2Seq model performance.

        :param valid_iterator: Valid or test iterator
        :type valid_iterator: torchtext.data.BucketIterator
        :return:
        """
        with torch.no_grad():
            total_loss = []
            text_size = self.config['text_size']
            for batch_id, batch in tqdm(enumerate(valid_iterator), total=len(valid_iterator), desc='Validation'):
                text, summary = self.get_text_summary_from_batch(batch)
                output = self.seq2seq(text, summary, teacher_forcing_ratio=0.0)
                loss = self.criterion(
                    output[1:].view(-1, text_size),
                    summary[1:].contiguous().view(-1),
                )
                total_loss.append(loss.data)
            return total_loss

    def get_text_summary_from_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtains original text and target summary indices from batch and transforms to GPU
        :param batch:
        :type batch: torchtext.data.batch.Batch
        :return: Text and summary indices for model
        :rtype: tuple
        """
        text = batch.text[0].to(get_device())
        summary = batch.summary[0].to(get_device())
        return text, summary

    def load_model(self, model_path: str, attention_param_path: str = None) -> None:
        """
        Loads trained model and transfers to GPU.
        Currently attention ``V`` parameter is also saved and loaded, cause PyTorch does not supports nn.Parameter
        saving directly.

        :param model_path: Path to trained model
        :type model_path: str
        :param attention_param_path: Path to trained attention parameter
        :type attention_param_path: str
        """
        if attention_param_path:
            self.seq2seq.load_state_dict(torch.load(model_path), strict=False)
            self.seq2seq.decoder.attention.v = nn.Parameter(torch.load(attention_param_path))
        else:
            self.seq2seq.load_state_dict(torch.load(model_path))
        self.seq2seq = self.seq2seq.to(get_device())

    def predict(self, text: str, length_of_original_text: float = 0.25) -> Tuple[str, torch.Tensor]:
        """
        Predicts model output / summarizes given text. Obtains summarization with defined maximum percentage of length
        of original text. Returns summarization and attention weights to plot attention heatmap.

        :param text: Original text to summarize
        :type text: str
        :param length_of_original_text: Maximum ratio of summary length comparing to original text
        :type length_of_original_text: float
        :return: summary text and attention weights
        :rtype: tuple
        """
        with torch.no_grad():
            sequence = self.vocab_config.indices_from_text(text).unsqueeze(0)
            sequence_length = sequence.size(1)
            encoder_outputs, encoder_hidden = self.encoder(sequence.transpose(0, 1))

            decoder_input = torch.LongTensor(
                [self.vocab_config.indices_from_text(Token.StartOfSequence.value)]).to(get_device())
            hidden = encoder_hidden[:self.decoder.n_layers]
            summary_words = [Token.StartOfSequence.value]
            max_summary_length = int(sequence_length * length_of_original_text)
            decoder_attentions = torch.zeros(max_summary_length, sequence_length)

            for idx in range(max_summary_length):
                output, hidden, decoder_attention = self.decoder(
                    decoder_input,
                    hidden,
                    encoder_outputs,
                )
                decoder_attentions[idx, :decoder_attention.size(2)] += \
                    decoder_attention.squeeze(0).squeeze(0).cpu().data
                top_v, top_i = output.data.topk(1)
                ni = top_i[0]
                if ni == self.vocab_config.indices_from_text(Token.EndOfSequence.value):
                    break
                else:
                    summary_words.append(self.vocab_config.text_from_indices(ni))

                decoder_input = torch.LongTensor([ni]).to(get_device())
            summary_words.append(Token.EndOfSequence.value)
            summary = " ".join(summary_words).lstrip()
            return summary, decoder_attentions

    def save_model(self, model_path: str, model_epoch: int) -> None:
        """
        Saves trained model weights after epoch, transferred to CPU.
        Currently attention ``V`` parameter is also saved, cause PyTorch does not supports nn.Parameter saving directly.

        :param model_path: Path to save model
        :type model_path: str
        :param model_epoch: Model epoch
        :type model_epoch: int
        """
        torch.save(self.seq2seq.cpu().state_dict(), model_path + f'_{model_epoch}.pt')
        torch.save(self.seq2seq.decoder.attention.v.cpu(), model_path + f'_att_param_{model_epoch}.pt')
        self.logger.info(f'Saved model {model_path}_{model_epoch}.pt')
        self.seq2seq.to(get_device())

    def show_loss(self, batch_id: int, loss: torch.Tensor, train_iterator: Any) -> None:
        """
        Logs loss value for specified batch.

        :param batch_id: Number of batch
        :type batch_id: int
        :param loss: Loss value for batch
        :param loss: torch.Tensor
        :param train_iterator: Train iterator
        :type train_iterator: torchtext.data.BucketIterator
        """
        self.logger.info(
            f'[{batch_id} / {len(train_iterator)}] Loss | {loss} | LR | {self.optimizer.param_groups[0]["lr"]}')
        if AVAILABLE_GPU:
            torch.cuda.empty_cache()

    def show_rouge_and_attention_matrix(self, epoch: int, batch_id: int, text: torch.Tensor, summary: torch.Tensor)\
            -> None:
        """
        Calls rouge metric calculation and attention heatmap drawing.

        :param epoch: Current training epoch.
        :type epoch: int
        :param batch_id: Number of batch
        :type batch_id: int
        :param text: Model input / original text indices tensor
        :type text: torch.Tensor
        :param summary: Model generated summary text indices tensor
        :type summary: torch.Tensor
        """
        original_text = self.vocab_config.text_from_indices(text.transpose(0, 1)[0])
        target_summary = self.vocab_config.text_from_indices(summary.transpose(0, 1)[0])
        output_summary, attention = self.predict(
            self.vocab_config.text_from_indices(text.transpose(0, 1)[0]))
        scores = calculate_rouge(hypothesis=output_summary, reference=target_summary)
        if scores:
            for key, value in scores[0].items():
                self.logger.info(
                    f'{key.upper()} [precision] : {np.round(value["p"] * 100, 2)} '
                    f'| [recall] : {np.round(value["r"] * 100, 2)} '
                    f'| [f-score] : {np.round(value["f"] * 100, 2)}',)
            draw_attention_matrix(
                attention=attention,
                original=original_text,
                summary=output_summary,
                config=self.config,
                epoch=epoch,
                batch_id=batch_id,
            )
        del original_text, target_summary, output_summary, attention, scores

    def train(self, train_iterator: Any, epoch: int = 0) -> List[torch.Tensor]:
        """
        Executes model training.

        :param train_iterator: Iterator over training dataset
        :type train_iterator: torchtext.data.BucketIterator
        :param epoch: Current training epoch.
        :type epoch: int
        :return: Training loss values for batches
        :rtype: list
        """
        grad_clip = self.config['grad_clip']
        text_size = self.config['text_size']
        self.seq2seq.train()
        total_loss = []
        for batch_id, batch in tqdm(enumerate(train_iterator), total=len(train_iterator), desc='Training'):
            text, summary = self.get_text_summary_from_batch(batch)
            self.optimizer.zero_grad()
            output = self.seq2seq(text, summary)
            loss = self.criterion(
                output[1:].view(-1, text_size),
                summary[1:].contiguous().view(-1),
            )
            if torch.isnan(loss):
                self.logger.info(f'NAN loss | output {output} | summary {summary} | text {text}')
                raise Exception

            loss.backward()
            clip_grad_norm_(self.seq2seq.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            total_loss.append(loss.data)

            if batch_id % 10 == 0:
                self.show_loss(batch_id, loss.data, train_iterator)

            if batch_id % 400 == 0:
                self.show_rouge_and_attention_matrix(epoch, batch_id, text, summary)
        return total_loss

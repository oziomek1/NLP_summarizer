import logging
import os
import spacy
import torch

from enum import Enum
from spacy.symbols import LEMMA, ORTH, POS
from typing import List

from nlper.file_io.reader import JsonReader
from nlper.utils.torch_utils import get_device


DEFAULT_VOCAB_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '../../resources/vocab_files/vocab.json'
)


class Token(Enum):
    Number = '<num>'
    Unknown = '<unk>'
    Padding = '<pad>'
    StartOfSentence = '<sos>'
    EndOfSentence = '<eos>'


class LangUtils:
    def __init__(self):
        self.logger = logging.getLogger(LangUtils.__name__)
        self.lang_model = None
        self.special_case = [{
            POS: 'NOUN',
            ORTH: Token.Number.value,
            LEMMA: Token.Number.value,
        }]

    def set_language_model(self, spacy_lang: str = 'pl_spacy_model', disable_options=None) -> spacy:
        try:
            self.lang_model = spacy.load(spacy_lang, disable=disable_options if disable_options else [])
            self.logger.info(f'Language model using SpaCy `pl_spacy_model`')
        except OSError as e:
            self.lang_model = spacy.load('en', disable=disable_options if disable_options else [])
            self.logger.warning(f'Language model SpaCy en : {e}')
        self.lang_model.tokenizer.add_special_case(Token.Number.value, self.special_case)
        return self.lang_model

    def tokenize_text(self, text: str) -> List[str]:
        return [tok.text for tok in self.lang_model.tokenizer(text)]


class VocabConfig:
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi
        self.itos = itos

    def set_vocab_from_field(self, text):
        self.stoi = text.vocab.stoi
        self.itos = text.vocab.itos

    def set_vocab_from_file(self, filepath=None):
        if not filepath:
            filepath = DEFAULT_VOCAB_CONFIG_PATH
        vocab = JsonReader().open_file(filepath=filepath)
        self.stoi = vocab['stoi']
        self.itos = vocab['itos']

    def indices_from_text(self, text: str) -> torch.Tensor:
        indices = [
            self.stoi.get(word, self.stoi.get(Token.Unknown.value))
            for word in text.strip().split(' ')
        ]
        return torch.LongTensor(indices).to(get_device())

    def text_from_indices(self, indices: List[int]) -> str:
        text = ""
        try:
            for element in indices:
                text += self.itos[element.item()] + " "
        except IndexError as e:
            print(e)
        return text

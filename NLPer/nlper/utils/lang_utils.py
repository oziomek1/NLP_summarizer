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
    """
    Language tokens representing

    * <num> - any numerical value including date and time in different formats
    * <unk> - unknown / out of vocabulary word
    * <pad> - padding for short text sequences in batch
    * <sos> - start of sequence for the beginning of summary generation
    * <eos> - end of sequence for marking the end of generated summary
    """
    Number = '<num>'
    Unknown = '<unk>'
    Padding = '<pad>'
    StartOfSequence = '<sos>'
    EndOfSequence = '<eos>'


class LangUtils:
    """
    Utils for SpaCy language model

    By default defines special case list of tokens to tokenizer.
    """
    def __init__(self):
        self.logger = logging.getLogger(LangUtils.__name__)
        self.lang_model = None
        self.special_case = [{
            POS: 'NOUN',
            ORTH: Token.Number.value,
            LEMMA: Token.Number.value,
        }]

    def set_language_model(self, spacy_lang: str = 'pl_spacy_model', disable_options=None) -> spacy:
        """
        Loads the SpaCy language model and adds the special case to tokenizer.
        By default tries to load spacy polish model and english model if first one is not available.

        :param spacy_lang: Name of language model
        :type spacy_lang: str
        :param disable_options: List of SpaCy options to disable, for example 'NER' for accelerated text parsing
        :return: Loaded Spacy language model
        :rtype: spacy
        """
        try:
            self.lang_model = spacy.load(spacy_lang, disable=disable_options if disable_options else [])
            self.logger.info(f'Language model using SpaCy `pl_spacy_model`')
        except OSError as e:
            self.lang_model = spacy.load('en', disable=disable_options if disable_options else [])
            self.logger.warning(f'Language model SpaCy en : {e}')
        self.lang_model.tokenizer.add_special_case(Token.Number.value, self.special_case)
        return self.lang_model

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenizes text using SpaCy language model

        :param text: Text to tokenize
        :type text: str
        :return: List of tokens
        :rtype: list
        """
        return [tok.text for tok in self.lang_model.tokenizer(text)]


class VocabConfig:
    """
    Utils for vocabulary

    :param stoi: Dictionary with text token and assigned index
    :type stoi: defaultdict, optional
    :param itos: List of text tokens
    :type itos: list, optional
    """
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi
        self.itos = itos

    def set_vocab_from_field(self, text: object) -> None:
        """
        Assigns dictionary of text tokens and indices, together with list of tokens from torchtext vocabulary.

        :param text: Torchtext vocabulary
        :param text: torchtext.Vocab
        """
        self.stoi = text.vocab.stoi
        self.itos = text.vocab.itos

    def set_vocab_from_file(self, filepath: str = None) -> None:
        """
        Loads vocabulary from file.

        :param filepath: Path to file with vocabulary
        :type filepath: str
        """
        if not filepath:
            filepath = DEFAULT_VOCAB_CONFIG_PATH
        vocab = JsonReader().open_file(filepath=filepath)
        self.stoi = vocab['stoi']
        self.itos = vocab['itos']

    def indices_from_text(self, text: str) -> torch.Tensor:
        """
        Converts text token to tensor of corresponding indices.

        :param text: Text to convert:
        :type text: str
        :return: Tensor with indices
        :rtype: torch.Tensor
        """
        indices = [
            self.stoi.get(word, self.stoi.get(Token.Unknown.value))
            for word in text.strip().split(' ')
        ]
        return torch.LongTensor(indices).to(get_device())

    def text_from_indices(self, indices: List[int]) -> str:
        """
        Converts list of indices into corresponding text - sequence of tokens.

        :param indices: List of indices
        :type indices: list
        :return: Text from tokens
        :rtype: str
        """
        text = ""
        try:
            for element in indices:
                text += self.itos[element.item()] + " "
        except IndexError as e:
            print(e)
        return text

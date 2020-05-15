import logging

from typing import Any
from typing import Dict
from typing import Tuple
from torchtext.data import BucketIterator
from torchtext.data import Dataset
from torchtext.data import Field
from torchtext.data import TabularDataset

from nlper.utils.lang_utils import LangUtils
from nlper.utils.lang_utils import Token


class DataLoader:
    """
    Data loader for model.
    Starts by initializing language utils. By default specifies disabled SpaCy language model options are
    ``ner`` and ``parser`` which significantly accelerates model training.

    :param config: Data loader config
    :type config: dict
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(DataLoader.__name__)
        self.config = config
        self.langUtils = LangUtils()
        self.disabled_language_options = ['ner', 'parser']
        self.iterators = None
        self.TEXT = None
        self.SUMMARY = None

    def load(self) -> Tuple[Tuple[BucketIterator, BucketIterator, BucketIterator], Field, Field]:
        """
        Calls language generation and data iterators creation for data loader.

        :return: Dataset iterators and fields with vocabulary
        :rtype: tuple
        """
        self.set_language()
        self.prepare_fields()
        self.load_splits_and_iterators()
        self.logger.info(f'Length of vocabulary {self.config["text_size"]}')
        return self.iterators, self.TEXT, self.SUMMARY

    def build_vocab(self, dataset: TabularDataset) -> None:
        """
        Builds vocabulary on dataset with definined special tokens and word frequency.
        The frequency is a minimum number of times a word must appear in dataset, to be placed into vocabulary.

        :param dataset: Dataset to build vocabulary on
        :type dataset: torchtext.TabularDataset
        """
        self.TEXT.build_vocab(
            dataset, specials=[Token.Number.value], min_freq=self.config['min_frequency_of_words_in_vocab'])
        self.SUMMARY.vocab = self.TEXT.vocab
        self.config['text_size'] = len(self.TEXT.vocab.itos)

    def load_iterators(self, splits: Tuple[Dataset, Dataset, Dataset]) -> None:
        """
        Obtains iterators for train, test, valid splits using torchtext BucketIterator

        :param splits: Tuple of tabular datasets for particular type of split
        :type splits: tuple
        """
        self.iterators = BucketIterator.splits(
            datasets=splits,
            batch_size=self.config['batch_size'],
            repeat=False,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False
        )

    def load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Loads train, test, valid splits using torchtext TabularDataset feature with

        :return:
        """
        return TabularDataset.splits(
            skip_header=True,
            path=self.config['train_test_val_dir'], format='csv',
            fields=[
                (self.config['dataframes_field_names'][0], self.TEXT),
                (self.config['dataframes_field_names'][1], self.SUMMARY),
            ],
            train='train.csv', validation='val.csv', test='test.csv'
        )

    def load_splits_and_iterators(self) -> None:
        """
        Calls train, test and valid split loading; vocabulary initialization and loading data iterators.

        """
        train, valid, test = self.load_splits()
        self.build_vocab(dataset=train)
        self.load_iterators(splits=(train, valid, test))

    def prepare_fields(self) -> None:
        """
        Initializes torchtext Fields with base token and language
        """
        self.TEXT = Field(tokenize=str.split, include_lengths=True, tokenizer_language='pl',
                          init_token=Token.StartOfSequence.value, eos_token=Token.EndOfSequence.value)
        self.SUMMARY = Field(tokenize=str.split, include_lengths=True, tokenizer_language='pl',
                             init_token=Token.StartOfSequence.value, eos_token=Token.EndOfSequence.value)

    def set_language(self) -> None:
        """
        Sets SpaCy language model to Polish with disables computation expensive language options.

        Default disabled language options
        * ``ner``
        * ``parser``
        """
        self.langUtils.set_language_model(disable_options=self.disabled_language_options)

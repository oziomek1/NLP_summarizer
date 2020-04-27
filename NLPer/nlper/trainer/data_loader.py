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
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(DataLoader.__name__)
        self.config = config
        self.langUtils = LangUtils()
        self.disabled_language_options = ['ner', 'parser']
        self.iterators = None
        self.TEXT = None
        self.SUMMARY = None

    def load(self) -> Tuple[Tuple[BucketIterator, BucketIterator, BucketIterator], Field, Field]:
        self.set_language()
        self.prepare_fields()
        self.load_splits_and_iterators()
        self.logger.info(f'Length of vocabulary {self.config["text_size"]}')
        return self.iterators, self.TEXT, self.SUMMARY

    def build_vocab(self, dataset) -> None:
        self.TEXT.build_vocab(
            dataset, specials=[Token.Number.value], min_freq=self.config['min_frequency_of_words_in_vocab'])
        self.SUMMARY.vocab = self.TEXT.vocab
        self.config['text_size'] = len(self.TEXT.vocab.itos)

    def load_iterators(self, splits: Tuple[Dataset, Dataset, Dataset]) -> None:
        self.iterators = BucketIterator.splits(
            datasets=splits,
            batch_size=self.config['batch_size'],
            repeat=False,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False
        )

    def load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
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
        train, valid, test = self.load_splits()
        self.build_vocab(dataset=train)
        self.load_iterators(splits=(train, valid, test))

    def prepare_fields(self) -> None:
        self.TEXT = Field(tokenize=str.split, include_lengths=True, tokenizer_language='pl',
                          init_token=Token.StartOfSentence.value, eos_token=Token.EndOfSentence.value)
        self.SUMMARY = Field(tokenize=str.split, include_lengths=True, tokenizer_language='pl',
                             init_token=Token.StartOfSentence.value, eos_token=Token.EndOfSentence.value)

    def set_language(self) -> None:
        self.langUtils.set_language_model(disable_options=self.disabled_language_options)

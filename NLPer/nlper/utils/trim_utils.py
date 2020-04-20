import logging
import numpy as np

from typing import Any
from typing import Callable
from typing import List

from nlper.utils.lang_utils import LangUtils


class TrimUtils:
    def __init__(self):
        self.logger = logging.getLogger(TrimUtils.__name__)
        self.lang_model = None

    def get_language_model(self) -> None:
        self.lang_model = LangUtils().set_language_model()

    def get_parsed_test(self, text: str) -> Any:
        if self.lang_model is None:
            self.get_language_model()
        return self.lang_model(text)

    def trim_text_to_upper_length_threshold(self, text: str, threshold: int) -> str:
        parsed_text = self.get_parsed_test(text=text)
        sentences_lengths = self.calculate_cumulative_sentences_lengths(list(parsed_text.sents))
        if sentences_lengths[-1] <= threshold:
            trimmed = list(parsed_text.sents)
        else:
            index = self.get_last_sentence_index(lengths=sentences_lengths, threshold=threshold)
            trimmed = self.trim_sentences(list(parsed_text.sents), index)
        joined = self.join_sentences(trimmed)
        return joined

    @staticmethod
    def calculate_cumulative_sentences_lengths(sentences: List[Any]) -> List[int]:
        return np.cumsum([sentence.__len__() for sentence in sentences])

    @staticmethod
    def get_last_sentence_index(lengths: List[int], threshold: int):
        last_sentence_index = list(map(lambda x: x > threshold, lengths)).index(True) - 1
        return last_sentence_index

    @staticmethod
    def join_sentences(sentences: List[Any]) -> str:
        return " ".join([token.text for token in sentences])

    @staticmethod
    def remove_text_below_lower_length_threshold(threshold: int) -> Callable:
        return lambda x: threshold < len(x.strip().split())

    @staticmethod
    def trim_sentences(sentences: List[str], index: int):
        return sentences[:index]

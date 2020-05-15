import logging
import numpy as np

from typing import Any
from typing import Callable
from typing import List

from nlper.utils.lang_utils import LangUtils


class TrimUtils:
    """
    Utils for trimming text to specified length
    """
    def __init__(self):
        self.logger = logging.getLogger(TrimUtils.__name__)
        self.lang_model = None

    def get_language_model(self) -> None:
        """
        Obtains the language model.
        """
        self.lang_model = LangUtils().set_language_model()

    def get_parsed_text(self, text: str) -> Any:
        """
        Parses text through language model from SpaCy.

        :param text: String text to be parsed
        :type text: str
        :return: SpaCy parsed text
        :rtype: spacy.tokens
        """
        if self.lang_model is None:
            self.get_language_model()
        return self.lang_model(text)

    def trim_text_to_upper_length_threshold(self, text: str, threshold: int) -> str:
        """
        Trims text to specified maximum length threshold. If text length is above the threshold, removes the last
        sentences until the total length does not exceed the threshold value. If the text length is below threshold,
        leaves the text unchanged.

        :param text: Text to be trimmed
        :type text: str
        :param threshold: Maximum length threshold
        :type threshold: int
        :return:
        """
        parsed_text = self.get_parsed_text(text=text)
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
        """
        Calculates cumulative length of sentences.

        :param sentences: List of sentences
        :type sentences: list
        :return: List of cumulative lengths
        :rtype: np.array
        """
        return np.cumsum([sentence.__len__() for sentence in sentences])

    @staticmethod
    def get_last_sentence_index(lengths: List[int], threshold: int):
        """
        Obtains index of last sentence in sequence before trimming.

        :param lengths: List of sentence lengths
        :type lengths: list
        :param threshold: Text sequence length threshold
        :type threshold: int
        :return: Index of last sentence
        :rtype: int
        """
        last_sentence_index = list(map(lambda x: x > threshold, lengths)).index(True) - 1
        return last_sentence_index

    @staticmethod
    def join_sentences(sentences: List[Any]) -> str:
        """
        Joins list of sentences into single text.

        :param sentences: List of sentences
        :type sentences: list
        :return: Joined text
        :rtype: str
        """
        return " ".join([token.text for token in sentences])

    @staticmethod
    def remove_text_below_lower_length_threshold(threshold: int) -> Callable:
        """
        Calls lambda function to check if sequence length is above specified length threshold.

        :param threshold: Length threshold value
        :type threshold: int
        :return: Calls anonymous lambda function
        :rtype: callable
        """
        return lambda x: threshold < len(x.strip().split())

    @staticmethod
    def trim_sentences(sentences: List[str], index: int):
        """
        Trims sequence of sentences to particular index.

        :param sentences: List of sentences
        :type sentences: list
        :param index: Index to trim at
        :type index: int
        :return: Trimmed list of sentences
        :rtype: list
        """
        return sentences[:index]

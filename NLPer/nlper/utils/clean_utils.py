import logging
import re

from bs4 import BeautifulSoup
from typing import List
from typing import Tuple

from nlper.utils.lang_utils import LangUtils


class CleanUtils:
    def __init__(self):
        self.logger = logging.getLogger(CleanUtils.__name__)
        self.lang_model = None

    def get_language_model(self) -> None:
        self.lang_model = LangUtils().set_language_model()

    def lemmatize(self, text: str) -> str:
        if self.lang_model is None:
            self.get_language_model()
        parsed_text = self.lang_model(text)
        return " ".join([
            sentence.lemma_ for sentence in parsed_text.sents
        ])

    @staticmethod
    def hide_numbers(text: str, number_replacement: str = '<num>') -> str:
        # Replace numbers with `<num>`
        text = re.sub(
            r"([0-9]+[.|,|;|:|/|\-|\\|MDCLXVI0-9]+)+|([0-9]+?)", " " + number_replacement + " ", str(text))
        return re.sub("\\s+", " ", str(text))

    @staticmethod
    def remove_hmtl_elements(text: str) -> str:
        return BeautifulSoup(text, features="html.parser").get_text(strip=True)

    @staticmethod
    def remove_non_text_chacters(text: str) -> str:
        text = re.sub("([\t\r\n])", ' ', str(text))  # removes tab and new line char
        text = re.sub("({.*\\})|(\\[.*\\])", '', str(text))  # removes curly braces and brackets with text inside
        text = re.sub("(__+|-+|~+|\\++|\\.\\.+|\\:+|\\/)", ' ', str(text))  # removes _ - ~ + .. : / chars
        text = re.sub("\\s+", ' ', str(text))  # removes whitespaces
        # removes various non-standard characters
        text = re.sub(
            r"[<>()|&©ø,;~*\[\]\'\"\`\\\"\„\”\“\‟\‶\‚\’\‘\‛\⁏\;\-\—\―\–\⁋\‰\\\%\^\&\*\$\#\@\!]", '', str(text))
        text = re.sub(r"[?!]", '.', str(text))  # replaces ? and ! chars with dot
        return text

    @staticmethod
    def remove_special_characters(text: str, characters: Tuple[str] = ('\\xao',)) -> str:
        for character in characters:
            if character in text:
                text = text.replace(character, ' ')
        return text

    @staticmethod
    def remove_characters_for_text(text: str) -> str:
        text = CleanUtils.remove_hmtl_elements(text)
        text = CleanUtils.remove_special_characters(text)
        text = CleanUtils.remove_non_text_chacters(text)
        return text

    @staticmethod
    def convert_list_to_text(text_as_list: List[str]) -> str:
        return " ".join(filter(None, text_as_list))

import logging
import re

from bs4 import BeautifulSoup
from typing import List
from typing import Tuple

from nlper.utils.lang_utils import LangUtils


class CleanUtils:
    """
    Utils for cleaning text
    """
    def __init__(self):
        self.logger = logging.getLogger(CleanUtils.__name__)
        self.lang_model = None

    def get_language_model(self) -> None:
        """
        Obtains the language model.
        """
        self.lang_model = LangUtils().set_language_model()

    def lemmatize(self, text: str) -> str:
        """
        Lemmatizes text using language model from SpaCy.

        :param text: String text to be lemmatized
        :type text: str
        :return: Lemmatized text
        :rtype: str
        """
        if self.lang_model is None:
            self.get_language_model()
        parsed_text = self.lang_model(text)
        return " ".join([
            sentence.lemma_ for sentence in parsed_text.sents
        ])

    @staticmethod
    def hide_numbers(text: str, number_replacement: str = '<num>') -> str:
        """
        Hides numbers, date and time in text with specified token. Supports various formats of numbers dates and times.

        Supported formats example:
        * 22 -> <num>
        * 11:45 -> <num>
        * 99.99 -> <num>
        * 596,789 -> <num>
        * 6;15 -> <num>
        * 5.99 -> <num>
        * 29/12/2010 -> <num>
        * 10\02\2000 -> <num>
        * 15.V.2030 -> <num>
        * 10.01.2020 -> <num>
        * 29/XII/1990 -> <num>
        * 22---22-2222 -> <num>
        * 22......22.2000 -> <num>
        * 01.01.2000 12:15 -> <num> <num>

        :param text: Text to hide numbers in
        :type text: str
        :param number_replacement: Token to replace numbers with
        :type number_replacement: str
        :return: Text with replaced numbers
        :rtype: str
        """
        text = re.sub(
            r"([0-9]+[.|,|;|:|/|\-|\\|MDCLXVI0-9]+)+|([0-9]+?)", " " + number_replacement + " ", str(text))
        return re.sub("\\s+", " ", str(text))

    @staticmethod
    def remove_hmtl_elements(text: str) -> str:
        """
        Removes html elements from text.

        Example:
        <p>sample text</p> -> sample text

        :param text: Text to remove html elements from
        :type text: str
        :return: Text with removed html elements
        :rtype: str
        """
        return BeautifulSoup(text, features="html.parser").get_text(strip=True)

    @staticmethod
    def remove_non_text_characters(text: str) -> str:
        """
        Removes non text characters from text.

        Types of characters to remove:
        * tab and new line characters - \t\r\n
        * curly braces and brackets with text inside - {}, []
        * characters: _ - ~ + .. : /
        * unnecessary whitespaces
        * various non-standard characters: <>()|&©ø,;~*'"`"„”“‟‶‚’‘‛⁏;- — ― –⁋%^‰&*$#@!

        Types of characters to replace:
        * ?! to be replaced with dot .

        :param text: Text to remove non text characters from
        :type text: str
        :return: Text with removed characters
        :rtype: str
        """
        text = re.sub("([\t\r\n])", ' ', str(text))
        text = re.sub("({.*\\})|(\\[.*\\])", '', str(text))
        text = re.sub("(__+|-+|~+|\\++|\\.\\.+|\\:+|\\/)", ' ', str(text))
        text = re.sub("\\s+", ' ', str(text))
        text = re.sub(
            r"[<>()|&©ø,;~*\[\]\'\"\`\\\"\„\”\“\‟\‶\‚\’\‘\‛\⁏\;\-\—\―\–\⁋\‰\\\%\^\&\*\$\#\@\!]", '', str(text))
        text = re.sub(r"[?!]", '.', str(text))
        return text

    @staticmethod
    def remove_special_characters(text: str, characters: Tuple[str] = ('\\xao',)) -> str:
        """
        Removes special characters.

        Example:
        * xao

        :param text: Text to remove special characters from
        :type text: str
        :param characters: Special characters to remove
        :type characters: tuple
        :return: Text with removed characters
        :rtype: str
        """
        for character in characters:
            if character in text:
                text = text.replace(character, ' ')
        return text

    @staticmethod
    def remove_characters_for_text(text: str) -> str:
        """
        Executes removal of various types of unwanted characters from text.

        :param text: Text to remove characters from
        :type text: str
        :return: Text with removed characters
        :rtype: str
        """
        text = CleanUtils.remove_hmtl_elements(text)
        text = CleanUtils.remove_special_characters(text)
        text = CleanUtils.remove_non_text_characters(text)
        return text

    @staticmethod
    def convert_list_to_text(text_as_list: List[str]) -> str:
        """
        Converts list of texts to single string, filtering the empty texts.

        :param text_as_list: List of texts
        :type text_as_list: list
        :return: Joined text
        :rtype: str
        """
        return " ".join(filter(None, text_as_list))

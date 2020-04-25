import logging

from nlper.utils.clean_utils import CleanUtils
from nlper.utils.lang_utils import VocabConfig


class Application:
    def __init__(self, text):
        self.text = text
        self.vocab_config = VocabConfig()
        self.logger = logging.getLogger(Application.__name__)
        self.clean_utils = CleanUtils()

    def run(self) -> None:
        self.clean_text()
        self.logger.info(f'Cleaned text | {self.text}')

    def remove_characters_and_hide_numbers(self) -> str:
        removed_character_text = self.clean_utils.remove_characters_for_text(text=self.text)
        return self.clean_utils.hide_numbers(text=removed_character_text)

    def lemmatize_text(self) -> None:
        self.text = self.clean_utils.lemmatize(text=self.text)

    def clean_text(self) -> None:
        self.text = self.remove_characters_and_hide_numbers()
        self.lemmatize_text()

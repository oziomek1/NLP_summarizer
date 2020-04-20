import logging

from nlper.utils.lang_utils import VocabConfig


class Application:
    def __init__(self, text):
        self.text = text
        self.vocab_config = VocabConfig()
        self.logger = logging.getLogger(Application.__name__)

    def clean(self):
        self.set_vocab()

    def set_vocab(self):
        if self.vocab_config.stoi is None and self.vocab_config.itos is None:
            self.vocab_config.set_vocab_from_file()
            self.logger.info(f'{self.vocab_config}')

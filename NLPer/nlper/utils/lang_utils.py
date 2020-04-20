import logging
import os
import spacy

from spacy.symbols import LEMMA, ORTH, POS

from nlper.file_io.reader import JsonReader


DEFAULT_VOCAB_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '../../resources/vocab_files/vocab.json'
)


class LangUtils:
    def __init__(self):
        self.logger = logging.getLogger(LangUtils.__name__)
        self.lang_model = None
        self.special_case = [{POS: 'NOUN', ORTH: '<num>', LEMMA: '<num>'}]

    def set_language_model(self, spacy_lang: str = 'pl_spacy_model', disable_options=None) -> spacy:
        try:
            self.lang_model = spacy.load(spacy_lang, disable=disable_options if disable_options else [])
            self.logger.info(f'Language model using SpaCy `pl_spacy_model`')
        except OSError as e:
            self.lang_model = spacy.load('en', disable=disable_options if disable_options else [])
            self.logger.warning(f'Language model SpaCy en : {e}')
        self.lang_model.tokenizer.add_special_case("<num>", self.special_case)
        return self.lang_model


class VocabConfig:
    def __init__(self, stoi=None, itos=None):
        self.set_default_tokens()
        self.stoi = stoi
        self.itos = itos

    def set_default_tokens(self):
        self.init_token = '<sos>'
        self.eos_token = '<eos>'
        self.num_token = '<num>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'

    def set_vocab_from_field(self, text):
        self.stoi = text.vocab.stoi
        self.itos = text.vocab.itos

    def set_vocab_from_file(self, filepath=None):
        if not filepath:
            filepath = DEFAULT_VOCAB_CONFIG_PATH
        vocab = JsonReader().open_file(filepath=filepath)
        self.stoi = vocab['stoi']
        self.itos = vocab['itos']

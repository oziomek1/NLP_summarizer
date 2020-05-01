import logging

from nlper.file_io.reader import JsonReader
from nlper.model.model import Model
from nlper.utils.clean_utils import CleanUtils
from nlper.utils.config_utils import read_config
from nlper.utils.lang_utils import Token
from nlper.utils.lang_utils import VocabConfig
from nlper.utils.train_utils import draw_attention_matrix


logging.basicConfig(
    format=f"%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s",
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


DEFAULT_PREDICT_CONFIG_PATH = 'resources/model_files/predict_config.yaml'


class Application:
    def __init__(self, text: str, config: str = DEFAULT_PREDICT_CONFIG_PATH):
        self.logger = logging.getLogger(Application.__name__)
        self.config = read_config(config, self.logger)
        self.text = text
        self.clean_utils = CleanUtils()
        self.json_reader = JsonReader()
        self.vocab_config = VocabConfig()
        self.model = None

    def run(self):
        self.prepare_vocab()
        self.prepare_text()
        self.prepare_model()
        self.predict()

    def predict(self):
        if self.config['length_of_original_text']:
            predicted, attention = self.model.predict(
                text=self.text,
                length_of_original_text=self.config['length_of_original_text'],
            )
        else:
            predicted, attention = self.model.predict(text=self.text)
        self.logger.info(f'Original : {self.text}')
        self.logger.info(f'Summary : {predicted}')
        draw_attention_matrix(attention=attention, original=self.text, summary=predicted)

    def prepare_text(self):
        def add_tokens(text) -> str:
            return f"{Token.StartOfSentence.value} {text} {Token.EndOfSentence.value}"

        def lemmatize_text(clean_utils, text) -> str:
            return clean_utils.lemmatize(text=text)

        def remove_characters_and_hide_numbers(clean_utils, text) -> str:
            removed_character_text = clean_utils.remove_characters_for_text(text=text)
            return clean_utils.hide_numbers(text=removed_character_text)

        self.text = remove_characters_and_hide_numbers(self.clean_utils, self.text)
        self.text = lemmatize_text(self.clean_utils, self.text)
        self.text = add_tokens(self.text)

    def prepare_model(self) -> None:
        self.model = Model(config=self.config, vocab_config=self.vocab_config)
        if self.config['use_dummy_model']:
            self.model.load_model(
                model_path=self.config['model_path'],
                attention_param_path=self.config['attention_param_path'],
            )
        self.logger.info(f'{self.model}')

    def prepare_vocab(self):
        self.vocab_config.__dict__ = self.json_reader.open_file(self.config['vocab_path'])
        self.config['text_size'] = len(self.vocab_config.itos)

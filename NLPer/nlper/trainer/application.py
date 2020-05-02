import logging
import os

from tqdm import tqdm

from nlper.file_io.writer import JsonWriter
from nlper.model.model import Model
from nlper.trainer.data_loader import DataLoader
from nlper.utils.config_utils import read_config
from nlper.utils.lang_utils import VocabConfig


logging.basicConfig(
    format=f"%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s",
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


class Application:
    def __init__(self, config: str):
        self.logger = logging.getLogger(Application.__name__)
        self.config = read_config(config, self.logger)
        self.vocab_config = VocabConfig()
        self.json_writer = JsonWriter()
        self.data_iterators = None
        self.TEXT = None
        self.SUMMARY = None
        self.model = None

    def run(self) -> None:
        self.data_iterators, self.TEXT, self.SUMMARY = DataLoader(config=self.config).load()
        self.prepare_and_save_vocab()
        self.prepare_model()
        self.train()

    def prepare_and_save_vocab(self):
        self.vocab_config.set_vocab_from_field(self.TEXT)
        self.json_writer.write(
            path=os.path.join(self.config['vocab_output_path'], self.config['model_name'] + '.json'),
            file={
                'itos': self.vocab_config.itos,
                'stoi': self.vocab_config.stoi,
            },
        )

    def prepare_model(self) -> None:
        self.model = Model(config=self.config, vocab_config=self.vocab_config)
        self.model.create_optimizers_and_loss()
        self.logger.info(f'{self.model}')

    def save_model(self, model_epoch: int) -> None:
        self.model.save_model(
            os.path.join(self.config['model_output_path'], self.config['model_name']), model_epoch)

    def train(self):
        train_iterator, valid_iterator, test_iterator = self.data_iterators
        best_loss = None
        for epoch in tqdm(range(1, self.config['epochs'] + 1)):
            self.model.train(train_iterator=train_iterator, epoch=epoch)
            valid_loss = self.model.evaluate(valid_iterator=valid_iterator)

            if not best_loss or valid_loss < best_loss:
                best_loss = valid_loss
                self.save_model(model_epoch=epoch)
        test_loss = self.model.evaluate(valid_iterator=test_iterator)
        self.logger.info(f'Test loss : {test_loss}')

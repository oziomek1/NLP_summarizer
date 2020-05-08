import logging
import os
import pandas as pd

from tqdm import tqdm
from typing import List

from nlper.file_io.writer import CsvWriter
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
        self.csv_writer = CsvWriter()
        self.json_writer = JsonWriter()
        self.data_iterators = None
        self.TEXT = None
        self.SUMMARY = None
        self.model = None
        self.start_epoch = 1

    def run(self) -> None:
        self.data_iterators, self.TEXT, self.SUMMARY = DataLoader(config=self.config).load()
        self.prepare_and_save_vocab()
        self.prepare_model()
        self.load_trained_model()
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

    def save_loss(self, loss: List[float], name: str, epoch: int = 0) -> None:
        self.csv_writer.write(
            path=os.path.join(self.config['model_output_path'], self.config['model_name'] + f'loss_{name}_{epoch}.csv'),
            file=pd.Series(loss),
        )

    def load_trained_model(self) -> None:
        if self.config['fine_tune_epoch'] is not None:
            loaded_epoch = self.config['fine_tune_epoch']
            path = os.path.join(self.config['model_output_path'], self.config['model_name'])
            model_path = path + f'_{loaded_epoch}.pt'
            attention_param_path = path + f'_att_param_{loaded_epoch}.pt'

            self.model.load_model(model_path=model_path, attention_param_path=attention_param_path)

            self.logger.info(f'Starting from {loaded_epoch} epoch | {model_path}')
            self.start_epoch += loaded_epoch

    def train(self):
        train_iterator, valid_iterator, test_iterator = self.data_iterators
        best_loss = None

        for epoch in tqdm(range(self.start_epoch, self.config['epochs'] + 1)):
            train_loss = self.model.train(train_iterator=train_iterator, epoch=epoch)
            valid_loss = self.model.evaluate(valid_iterator=valid_iterator)

            best_loss = sum(valid_loss) / len(valid_loss)
            self.save_model(model_epoch=epoch)

            self.save_loss(loss=train_loss, name='train', epoch=epoch)
            self.save_loss(loss=valid_loss, name='valid', epoch=epoch)

        test_loss = self.model.evaluate(valid_iterator=test_iterator)
        self.logger.info(f'Test loss : {test_loss}')
        self.save_loss(loss=test_loss, name='test')

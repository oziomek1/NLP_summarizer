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
    """
    Model train application.
    Starts by initializing vocabulary config and writers for saving updated vocabulary and loss function results.
    By default model training starts with epoch number equal to 1, can be altered while fine tuning the model.

    :param config_path: Path to yaml config file, by default loads example config
    :type config_path: str
    """
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(Application.__name__)
        self.config = read_config(config_path, self.logger)
        self.vocab_config = VocabConfig()
        self.csv_writer = CsvWriter()
        self.json_writer = JsonWriter()
        self.data_iterators = None
        self.TEXT = None
        self.SUMMARY = None
        self.model = None
        self.start_epoch = 1

    def run(self) -> None:
        """
        Executes model training process.
        """
        self.data_iterators, self.TEXT, self.SUMMARY = DataLoader(config=self.config).load()
        self.prepare_and_save_vocab()
        self.prepare_model()
        self.load_trained_model()
        self.train()

    def prepare_and_save_vocab(self) -> None:
        """
        Sets vocabulary from torchtext.Field and saves it to file.
        """
        self.vocab_config.set_vocab_from_field(self.TEXT)
        self.json_writer.write(
            path=os.path.join(self.config['vocab_output_path'], self.config['model_name'] + '.json'),
            file={
                'itos': self.vocab_config.itos,
                'stoi': self.vocab_config.stoi,
            },
        )

    def prepare_model(self) -> None:
        """
        Initializes model for training.
        Calls method to create optimizers and loss functions.
        """
        self.model = Model(config=self.config, vocab_config=self.vocab_config)
        self.model.create_optimizers_and_loss()
        self.logger.info(f'{self.model}')

    def save_model(self, model_epoch: int) -> None:
        """
        Calls method to save model after particular epoch.

        :param model_epoch: Number of training epoch to save model after.
        :type model_epoch: int
        """
        self.model.save_model(
            os.path.join(self.config['model_output_path'], self.config['model_name']), model_epoch)

    def save_loss(self, loss: List[float], name: str, epoch: int = 0) -> None:
        """
        Calls method to save model losses using particular loss type and after particular epoch.

        :param loss: List with loss function values.
        :type loss: list
        :param name: Loss function type name, for example: train or text
        :type name: str
        :param epoch: Number of training epoch to save loss after.
        :type epoch: int
        """
        self.csv_writer.write(
            path=os.path.join(self.config['model_output_path'], self.config['model_name'] + f'loss_{name}_{epoch}.csv'),
            file=pd.Series(loss),
        )

    def load_trained_model(self) -> None:
        """
        Calls method to load model parameters for particular epoch for fine tuning.
        Also alters the next epoch number to continue training.
        """
        if self.config['fine_tune_epoch'] is not None:
            loaded_epoch = self.config['fine_tune_epoch']
            path = os.path.join(self.config['model_output_path'], self.config['model_name'])
            model_path = path + f'_{loaded_epoch}.pt'
            attention_param_path = path + f'_att_param_{loaded_epoch}.pt'

            self.model.load_model(model_path=model_path, attention_param_path=attention_param_path)

            self.logger.info(f'Starting from {loaded_epoch} epoch | {model_path}')
            self.start_epoch += loaded_epoch

    def train(self) -> None:
        """
        Performs training and evaluation of model having train, valid and test iterators.
        Calls saving model and loss after every epoch.
        """
        train_iterator, valid_iterator, test_iterator = self.data_iterators

        for epoch in tqdm(range(self.start_epoch, self.config['epochs'] + 1)):
            train_loss = self.model.train(train_iterator=train_iterator, epoch=epoch)
            valid_loss = self.model.evaluate(valid_iterator=valid_iterator)

            self.save_model(model_epoch=epoch)

            self.save_loss(loss=train_loss, name='train', epoch=epoch)
            self.save_loss(loss=valid_loss, name='valid', epoch=epoch)

        test_loss = self.model.evaluate(valid_iterator=test_iterator)
        self.logger.info(f'Test loss : {test_loss}')
        self.save_loss(loss=test_loss, name='test')

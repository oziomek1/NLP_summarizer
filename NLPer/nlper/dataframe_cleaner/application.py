import logging

from nlper.dataframe_cleaner.reducer import Reducer
from nlper.dataframe_cleaner.cleaner import Cleaner
from nlper.dataframe_cleaner.trimmer import Trimmer
from nlper.file_io.dataframe_reader import FileReader
from nlper.file_io.dataframe_writer import FileWriter
from nlper.utils.lang_utils import LangUtils
from nlper.utils.config_utils import read_config


logging.basicConfig(
    format=f"%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s",
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


class Application:
    """
    Data frame cleaner application, starts by initializing read and write objects.

    :param config_path: Text to clean
    :type config_path: str
    """
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(Application.__name__)
        self.config = read_config(config_path, self.logger)
        self.file_reader = FileReader(path=self.config['input'])
        self.file_writer = FileWriter(path=self.config['output'])
        self.data = None

    def run(self) -> None:
        """
        Executes data frame cleaning process.
        """
        self.read_files()
        self.reduce_dataframes()
        self.load_language_model()
        self.clean_dataframes()
        self.trim_dataframes()

    def clean_dataframes(self) -> None:
        """
        Calls text in data frame cleaning of every data frame using cleaner.
        Saves cleaned data frame if specified in a config file.
        """
        for name, value in self.data.items():
            self.logger.info(f'Cleaning : {name} data : {len(value)}')
            self.data[name] = Cleaner(config=self.config, data=value).clean_dataframe()
        self.check_if_should_save(type='cleaned')

    def check_if_should_save(self, type: str) -> None:
        """
        Resolves data frame saving after particular procedure, based on config file.
        :param type: Name of procedure to save after
        :type type: str
        """
        if self.config[f'save_{type}']:
            self.logger.info(f'Saving {type} data')
            self.file_writer.save_file(
                data=self.data,
                name=self.config[f'{type}_output_name'],
                merge_data=self.config[f'{type}_merge_data'],
                output_type=self.config[f'{type}_output_type'],
            )

    def load_language_model(self) -> None:
        """
        Initializes and obtains the language model from SpaCy.
        """
        if self.config['lemmatize'] or self.config['trim_data']:
            lang_model = LangUtils()
            self.config['language_model'] = lang_model.set_language_model()

    def read_files(self) -> None:
        """
        Calls data frames reading procedure using file reader.
        """
        self.data = self.file_reader.read_json_lines_files()

    def reduce_dataframes(self) -> None:
        """
        Calls data frame reduction of every data frame using reducer.
        Saves reduced data frame if specified in a config file.
        """
        for name, value in self.data.items():
            self.data[name] = Reducer(config=self.config, data=value).reduce_dataframe()
        self.check_if_should_save(type='reduced')

    def trim_dataframes(self) -> None:
        """
        Calls text in data frame trimming of every data frame using trimmer.
        Saves reduced data frame if specified in a config file.
        """
        if self.config['trim_data']:
            for name, value in self.data.items():
                self.logger.info(f'Trimming : {name} data : {len(value)}')
                self.data[name] = Trimmer(config=self.config, data=value).trim_dataframe()
        self.check_if_should_save(type='trimmed')

import yaml

from typing import Any
from typing import Dict


def read_config(config_path: str, logger: Any = None) -> Dict:
    """
    Reads and returns config from yaml file
    :param config_path: Path to yaml config
    :type config_path: str
    :param logger: Logger, logs the loaded config
    :param logger: logger
    :return: Loaded config dictionary
    :rtype: dict
    """
    with open(config_path, 'r') as file:
        config = yaml.load(file, yaml.FullLoader)
    if logger:
        logger.info(f'{config_path} : {config}')
    return config

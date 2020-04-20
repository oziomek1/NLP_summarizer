import yaml

from typing import Any
from typing import Dict


def read_config(config_path: str, logger: Any = None) -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.load(file, yaml.FullLoader)
    if logger:
        logger.info(f'config_path{config}')
    return config

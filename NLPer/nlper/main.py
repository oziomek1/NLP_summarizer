import click
from nlper.exceptions import MissingFilePathOrConfigException


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config',
                required=True,
                type=click.Path(exists=True, file_okay=True, dir_okay=False))
def clean_data(config: str):
    """
    Clean data frames.

    :param config: Path to config file
    :type config: str
    """
    from nlper.dataframe_cleaner import main as dataframe_cleaner_app

    dataframe_cleaner_app(config=config)


@cli.command()
@click.argument('text')
def clean_text(text: str):
    """
    Clean single text.

    :param text: Text to clean
    :type text: str
    """
    from nlper.text_cleaner import main as text_cleaner_app

    text_cleaner_app(text=text)


@cli.command()
@click.argument('text')
def predict(text: str):
    """
    Generate summary for given text.

    :param text: Text to summarize
    :type text: str
    """
    from nlper.predictor import main as predictor_app

    predictor_app(text=text)


@cli.command()
@click.argument('config',
                default=None,
                required=False,
                type=click.Path(exists=True, dir_okay=False))
@click.option('--filepath', default=None, show_default=True)
@click.option('--valid', default=True, show_default=True)
def split_train_test(config, filepath, valid):
    """
    Split data frame into train, test and valid for training.

    :param config: Path to config file
    :type config: str
    :param filepath: Data frame file path
    :type filepath: str
    :param valid: Flag to include split into valid part
    :type valid: bool
    """
    if not config and not filepath:
        raise MissingFilePathOrConfigException()
    from nlper.utils import main as split_train_test_app

    split_train_test_app(config, filepath, valid)


@cli.command()
@click.argument('config',
                required=True,
                type=click.Path(exists=True, file_okay=True, dir_okay=False))
def train(config: str):
    """
    Train model.

    :param config: Path to config file
    :type config: str
    """
    from nlper.trainer import main as trainer_app

    trainer_app(config=config)


if __name__ == '__main__':
    cli()

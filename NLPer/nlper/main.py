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
    from nlper.dataframe_cleaner import main as dataframe_cleaner_app

    dataframe_cleaner_app(config=config)


@cli.command()
@click.argument('text')
def clean_text(text: str):
    from nlper.text_cleaner import main as text_cleaner_app

    text_cleaner_app(text=text)


@cli.command()
@click.argument('text')
def predict(text: str):
    pass


@cli.command()
@click.argument('config',
                default=None,
                required=False,
                type=click.Path(exists=True, dir_okay=False))
@click.option('--filepath', default=None, show_default=True)
@click.option('--valid', default=True, show_default=True)
def split_train_test(config, filepath, valid):
    if not config and not filepath:
        raise MissingFilePathOrConfigException()
    from nlper.utils import main as split_train_test_app

    split_train_test_app(config, filepath, valid)


@cli.command()
@click.argument('config',
                type=click.Path(exists=True, dir_okay=False))
def train(config: str):
    from nlper.trainer import main as trainer_app

    trainer_app(config=config)


if __name__ == '__main__':
    cli()

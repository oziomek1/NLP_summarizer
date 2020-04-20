import click


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
    print(f'predict {text}')
    pass


@cli.command()
@click.argument('config',
                type=click.Path(exists=True, dir_okay=False))
def train(config: str):
    pass


if __name__ == '__main__':
    cli()

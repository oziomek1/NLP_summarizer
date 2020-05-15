import sys

from nlper.dataframe_cleaner.application import Application


def main(config: str):
    """
    Executes the data frame cleaning pipeline.

    :param config: Path to config
    :type config: str
    """
    application = Application(config_path=config)
    application.run()


if __name__ == '__main__':
    main(sys.argv[1])

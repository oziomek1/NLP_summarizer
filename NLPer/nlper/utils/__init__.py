import sys

from nlper.utils.train_test_splitter import TrainTestSplitter


def main(config: str, filepath: str, valid: bool):
    """
    Executes the train, test, valid split pipeline.

    :param config: Path to config
    :type config: str
    :param filepath: Path to data frame to split
    :type filepath: str
    :param valid: Flag to split into valid part
    :type valid: bool
    """
    application = TrainTestSplitter(config=config, filepath=filepath, valid=valid)
    application.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], bool(sys.argv[3]))

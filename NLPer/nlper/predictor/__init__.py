import sys

from nlper.predictor.application import Application


def main(text: str):
    """
    Executes the text prediction pipeline, obtaining the summarization

    :param text: Text to summarize
    :type text: str
    """
    application = Application(text=text)
    application.run()


if __name__ == '__main__':
    main(sys.argv[1])

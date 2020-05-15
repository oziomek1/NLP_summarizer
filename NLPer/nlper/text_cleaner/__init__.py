from nlper.text_cleaner.application import Application


def main(text: str):
    """
    Executes the text cleaning pipeline.

    :param text: Text to clean
    :type text: str
    """
    application = Application(text=text)
    application.run()


if __name__ == '__main__':
    main()

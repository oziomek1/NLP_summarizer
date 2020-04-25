from nlper.text_cleaner.application import Application


def main(text: str):
    application = Application(text=text)
    application.run()


if __name__ == '__main__':
    main()

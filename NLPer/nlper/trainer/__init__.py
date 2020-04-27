import sys

from nlper.trainer.application import Application


def main(config: str):
    application = Application(config=config)
    application.run()


if __name__ == '__main__':
    main(sys.argv[1])

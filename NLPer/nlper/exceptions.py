class NLPerException(Exception):
    """
    Base application exception
    """
    _template = 'NLPerException: {}'

    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return self._template.format(*self.args)


class UnsupportedFileTypeException(NLPerException):
    _template = 'Unable to handle {} file extension'


class MissingFilePathOrConfigException(NLPerException):
    _template = 'Provide config file or --filepath [FILEPATH] parameters'

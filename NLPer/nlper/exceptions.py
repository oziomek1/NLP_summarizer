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
    """
    Exception raised when the provided file is in an unsupported type.
    """
    _template = 'Unable to handle {} file extension'


class MissingFilePathOrConfigException(NLPerException):
    """
    Exception raised when missing both path to data file and config
    """
    _template = 'Provide config file or --filepath [FILEPATH] parameters'

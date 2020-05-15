import logging

from nlper.utils.clean_utils import CleanUtils


class Application:
    """
    Text cleaner application, starts by initializing object with cleaning utils.

    :param text: Text to clean
    :type text: str
    """
    def __init__(self, text):
        self.text = text
        self.logger = logging.getLogger(Application.__name__)
        self.clean_utils = CleanUtils()

    def run(self) -> None:
        """
        Executes text cleaning process.
        """
        self.clean_text()
        self.logger.info(f'Cleaned text | {self.text}')

    def remove_characters_and_hide_numbers(self) -> str:
        """
        Calls removing special characters and hiding numbers procedures using cleaning utils.

        * Removed characters includes html and non text chars.
        * Hidden numbers includes different number formats, dates and time.

        :return: Text without special characters and with hidden numbers
        :rtype: str
        """
        removed_characters_text = self.clean_utils.remove_characters_for_text(text=self.text)
        return self.clean_utils.hide_numbers(text=removed_characters_text)

    def lemmatize_text(self) -> None:
        """
        Calls text lemmatization procedure using cleaning utils.
        """
        self.text = self.clean_utils.lemmatize(text=self.text)

    def clean_text(self) -> None:
        """
        Calls cleaning operations on given text.
        """
        self.text = self.remove_characters_and_hide_numbers()
        self.lemmatize_text()

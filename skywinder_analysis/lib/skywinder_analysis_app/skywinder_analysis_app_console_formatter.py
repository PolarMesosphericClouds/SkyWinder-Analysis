import logging


class SkyWinderAnalysisAppConsoleFormatter(logging.Formatter):
    '''
    A custom formatter for SkyWinder-Analysis apps designed to
    use more verbose output for errors and debug
    messages.

    Template taken from:
        LeapAppConsoleFormatter
        http://stackoverflow.com/questions/1343227/can-pythons-logging-format-be-modified-depending-on-the-message-log-level
    '''

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)
        self.error_format = '%(levelname)s: %(filename)s: %(funcName)s: %(message)s'
        self.debug_format = self.error_format
        self.critical_format = self.error_format
        self.warning_format = "%(levelname)s: %(message)s"
        self.info_format = "%(message)s"

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.ERROR:
            self._fmt = self.error_format
        elif record.levelno == logging.DEBUG:
            self._fmt = self.debug_format
        elif record.levelno == logging.CRITICAL:
            self._fmt = self.critical_format
        elif record.levelno == logging.WARNING:
            self._fmt = self.warning_format
        elif record.levelno == logging.INFO:
            self._fmt = self.info_format

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result

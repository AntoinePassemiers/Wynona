# -*- coding: utf-8 -*-
# exceptions.py
# author : Antoine Passemiers


class ContactMapException(Exception):
    pass


class EvaluationException(Exception):
    pass


class UnsupportedExtensionError(Exception):
    pass


class TextFileParsingError(Exception):
    pass


class NotImplementedMethodError(Exception):
    pass


class HyperParameterError(Exception):
    pass


class EarlyStoppingException(Exception):
    pass

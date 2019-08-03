# -*- coding: utf-8 -*-
# base.py: Base parser
# author : Antoine Passemiers

from wynona.parsers.exceptions import UnsupportedExtensionError

import os


class Parser:

    def getSupportedExtensions(self):
        return list()

    def parse(self, filepath):
        _, file_ext = os.path.splitext(filepath)
        if not file_ext in self.getSupportedExtensions():
            raise UnsupportedExtensionError(
                "Extension %s is not supported by parser %s" % (file_ext, self.__class__.__name__))
        else:
            return self.__parse__(filepath)

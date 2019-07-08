# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup


packages = [
        'wynona',
        'wynona.nn',
        'wynona.prot']

setup(
    name='wynona',
    version='1.0.0',
    description='Protein Contact Prediction based on a Fully-Convolution Neural Model',
    url='https://github.com/AntoinePassemiers/Wynona',
    author='Antoine Passemiers',
    author_email='apassemi@ulb.ac.be',
    packages=packages,
    include_package_data=False,
    install_requires=[
        'numpy >= 1.13.3',
        'scipy >= 1.1.0',
        'torch >= 1.0.0'])

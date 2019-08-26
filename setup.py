#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(name='Hilbert_Autoencoder',
      version='0.0.1',
      packages=find_packages(
          exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      description='Hilbert Autoencoder')

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:08:42 2022

@author: Yuchen Wang
"""

from setuptools import setup, find_packages

import pyttop

setup_kwargs = dict(
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
        # 'python>=3.8',
    ],
    python_requires='>=3.8',
    )

setup(
    name='pyttop',
    version=pyttop.__version__,
    packages=find_packages(include=['pyttop', 'pyttop.*']),
    **setup_kwargs,
)



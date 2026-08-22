#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

# Name, version and dependencies live in pyproject.toml ([project]); this file
# only pins the flat-layout package discovery for the editable install.
setup(
    packages=find_packages(exclude=("configs", "tests", "tools", "demo")),
)

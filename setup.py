from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

extensions = [
]

setup(
    name="pyspdetection",
    version=0.1,
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    setup_requires=[]
)
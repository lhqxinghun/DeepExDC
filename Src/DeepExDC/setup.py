import sys
from pathlib import Path
import setuptools
from setuptools import setup, Extension
import numpy

if sys.version_info < (3, 9):
    sys.exit('DeepExDC requires Python >= 3.9')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


distance_module = Extension(
    'pyvegdist',
    sources=['pyvegdist.c'],
    include_dirs=[numpy.get_include()]
)

setuptools.setup(
    name="DeepExDC",
    version="0.1.0",
    author="",
    author_email="",
    description="An interpretable one-dimensional convolutional neural network for differential analysis of A/B compartments in scHi-C data across multiple populations of individuals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lhqxinhun/DeepExDC",
    project_urls={
        "Bug Tracker": "https://github.com/lhqxinhun/DeepExDC",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    ext_modules=[distance_module],
)

import io
import os
import re
from setuptools import setup, find_packages

PACKAGE_NAME = "model_downstream_interface"

def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


VERSION = find_version(PACKAGE_NAME, '__init__.py')


def find_requirements(file_path):
    with open(file_path) as f:
        return f.read().splitlines()


requirements = find_requirements('requirements.txt')

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="An interface and utilities library for tranferring big model to downstream task.",
    url="",
    author="chenyaofo",
    author_email="chenyaofo@gmail.com",
    packages=find_packages(exclude=['tests']),
    package_data={'': ['requirements.txt']},
    install_requires=requirements,
    include_package_data=True,
)
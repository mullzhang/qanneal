import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='qanneal',
    version='0.0.3',
    description='Python library for simulation of quantum annealing',
    long_description='README.md',
    author='mullzhang',
    install_requires=read_requirements(),
    url='https://github.com/mullzhang/qanneal',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)

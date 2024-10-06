from setuptools import setup, find_packages

setup(
    name='mysupertools2',
    version='0.1',
    description='Module 2-Exercice 2',
    author='Hugus35',
    author_email='hugo.bonfils.1@etu.sorbonne-universite.fr',
    packages=find_packages(exclude=['mysupertools2.tests']),
    entry_points={
        'console_scripts': [
            'mysupertools2=mysupertools2.cli:main'
            ]
    },
    extras_require={
        'dev': ['pytest'],
    },
)
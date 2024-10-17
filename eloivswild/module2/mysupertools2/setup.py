from setuptools import setup, find_packages

setup(
    name='mysupertools2',
    version='0.2',
    packages=find_packages(exclude=['mysupertools2.tests']),
    install_requires=['pandas'],
    entry_points={
        'console_scripts': [
            'mysupertools2=mysupertools2.cli:main',
        ],
    },
    author= "EloixChatGPT",
    extras_require={
    'dev': ['pytest'],
},
)
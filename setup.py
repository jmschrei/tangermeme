from setuptools import setup

setup(
    name='tangermeme',
    version='0.0.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['tangermeme', 'tangermeme/tools'],
    scripts=[],
    url='https://github.com/jmschrei/tangermeme',
    license='LICENSE.txt',
    description='Biological sequence analysis for the modern age.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "pandas >= 1.3.3",
        "pyBigWig >= 0.3.17",
        "torch >= 1.9.0",
        "pyfaidx >= 0.7.2.1",
        "tqdm >= 4.64.1",
        "numba >= 0.55.1",
        "logomaker",
        "joblib >= 1.3.2"
    ],
)
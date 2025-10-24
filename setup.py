from setuptools import setup

setup(
    name='tangermeme',
    version='1.0.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['tangermeme'],
    #scripts=['cmd/tangermeme'],
    url='https://github.com/jmschrei/tangermeme',
    license='LICENSE.txt',
    description='Biological sequence analysis for the modern age.',
    install_requires=[
        "numpy >= 1.14.2, <= 2.0.1",
        "scipy >= 1.0.0",
        "pandas >= 1.3.3",
        "torch >= 1.9.0",
        "pybigtools",
        "pyfaidx >= 0.7.2.1",
        "tqdm >= 4.64.1",
        "numba >= 0.55.1",
        "logomaker",
        "joblib >= 1.3.2",
        "scikit-learn >= 1.2.2",
        "matplotlib",
        "memelite"
    ],
)

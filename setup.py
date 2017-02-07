from setuptools import setup, find_packages

setup(
    name='ml_helpers',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    long_description=open('README.rst').read(),
    author="Illarion Khlestov",
    author_email="khlyestovillarion@gmail.com",
    install_requires=[
       'numpy',
       'matplotlib',
    ]
)

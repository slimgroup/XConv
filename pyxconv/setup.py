from setuptools import setup, find_packages

setup(
    name = 'pyxconv',
    version = '0.1',
    author = 'Mathias Louboutin',
    author_email = 'mlouboutin3@gatech.edu',
    license = 'MIT',
    install_requires=['torch', 'opt_einsum'],
    packages = find_packages(),
   
)

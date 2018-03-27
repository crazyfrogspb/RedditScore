from setuptools import setup, find_packages
import sys
import os

if sys.version_info[0] != 3:
    raise RuntimeError('Unsupported python version "{0}"'.format(
      sys.version_info[0]))

try:
    with open('README.md') as f:
        long_description = f.read()
except:
    long_description = 'Package for performing Reddit-based text analysis'

setup(name='redditscore',
      version='0.1',
      description='Package for performing Reddit-based text analysis',
      author='Evgenii Nikitin',
      author_email='e.nikitin@nyu.edu',
      licence="MIT License",
      url='http://e.nikitin.com',
      packages=['redditscore'],
	  package_dir={'redditscore': 'redditscore'},
	  package_data={'redditscore': ['redditscore/data/*.*']},
	  long_description=long_description,
	  install_requires=[
          'spacy'
      ],
     )

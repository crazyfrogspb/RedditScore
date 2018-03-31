from setuptools import setup, find_packages
import sys
import os

if sys.version_info[0] != 3:
    raise RuntimeError('Unsupported python version "{0}"'.format(
      sys.version_info[0]))
try:
    with open('README.md') as f:
        readme = f.read()
except:
    readme = 'Package for performing Reddit-based text analysis'
	
INSTALL_REQUIRES = ["setuptools", "spacy>=2.0.10", "tldextract>=2.1.0", "requests>=2.18.4", "nltk>=3.2.5"]

with open('LICENSE') as f:
    license = f.read()

setup(
    name='redditscore',
    version='0.2.0',
    description='Package for performing Reddit-based text analysis',
    long_description=readme,
    author='Evgenii Nikitin',
    author_email='e.nikitin@nyu.edu',
    url='https://github.com/crazyfrogspb/RedditScore',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
	include_package_data=True,
	install_requires=INSTALL_REQUIRES,
)
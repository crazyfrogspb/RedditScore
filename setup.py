import os
import sys

from setuptools import find_packages, setup

if sys.version_info[0] != 3:
    raise RuntimeError('Unsupported python version "{0}"'.format(
        sys.version_info[0]))
try:
    with open('README.md') as f:
        readme = f.read()
except Exception:
    readme = 'Package for performing Reddit-based text analysis'


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    INSTALL_REQUIRES = ["setuptools", "spacy>=2.0.11", "tldextract>=2.1.0", "requests>=2.18.0",
                        "scikit-learn>=0.19.0", "pandas>=0.22.0", "scipy>=1.0.0", "numpy>=1.14.0",
                        "matplotlib>=2.2.0", "beautifulsoup4>=4.6.0", "adjustText>=0.6.3"]
else:
    INSTALL_REQUIRES = ["setuptools", "requests>=2.18.0"]

EXTRAS = {
    "nltk": ["nltk>=3.2.5"],
    "neural_nets": ["keras>=2.1.5"],
    "fasttext": ["fasttext"],
}


with open('LICENSE') as f:
    license = f.read()

setup(
    name='redditscore',
    version='0.5.1',
    description='Package for performing Reddit-based text analysis',
    long_description=readme,
    author='Evgenii Nikitin',
    author_email='e.nikitin@nyu.edu',
    url='https://github.com/crazyfrogspb/RedditScore',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,
    dependency_links=[
        'http://github.com/crazyfrogspb/fastText/tarball/master#egg=fasttext',
        'https://github.com/walmsley/tweepy.git@patch-1#egg=tweepy=3.6.0'],
)

from setuptools import setup

setup(
    name='twitter_processor',
    version='1.0',
    description='Python package to analyse Twitter database using natural language processing (NLP) related to sentiment analysis',
    author='José Mallent Trenor',
    author_email='jmallent@uoc.edu',
    packages=['twitter_processor'],
    install_requires=[
        'nltk',
        'wordcloud',
        'matplotlib',
        'unittest',
        'coverage'
    ],
)

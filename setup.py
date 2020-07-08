from setuptools import setup, find_packages

extras_requires = {
    "spacy": ["spacy>=2.1,<2.2"],
}

setup(
    name='bothub_nlp_rasa_utils',
    version='1.1.8',
    description='Bothub NLP Rasa Utils',
    packages=find_packages(),
    install_requires=[
        'rasa==1.10.6',
    ],
    extras_require=extras_requires,
)

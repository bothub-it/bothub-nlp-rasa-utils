from setuptools import setup, find_packages

extras_requires = {
    "spacy": ["spacy>=2.1,<2.2"],
}

setup(
    name='bothub_nlp_rasa_utils',
    version='1.1.28staging',
    description='Bothub NLP Rasa Utils',
    packages=find_packages(),
    package_data={'bothub_nlp_rasa_utils.lookup_tables': ['en/location.txt', 'en/email.txt', 'en/money.txt', 'pt_br/location.txt', 'pt_br/cep.txt', 'pt_br/cpf.txt', 'pt_br/email.txt', 'pt_br/money.txt']},
    install_requires=[
        'rasa==1.10.6',
        'transformers==2.11.0',
        'emoji==0.6.0',
        'recognizers-text-suite'
    ],
    extras_require=extras_requires,
)

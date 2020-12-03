from setuptools import setup, find_packages

extras_requires = {
    "spacy": ["spacy>=2.1,<2.2"],
}

setup(
    name='bothub_nlp_rasa_utils',
    version='1.1.32',
    description='Bothub NLP Rasa Utils',
    packages=find_packages(),
    package_data={'bothub_nlp_rasa_utils.lookup_tables': ['en/country.txt', 'en/email.txt', 'pt_br/country.txt', 'pt_br/cep.txt', 'pt_br/cpf.txt', 'pt_br/email.txt', 'pt_br/brand.txt']},
    install_requires=[
        'rasa==1.10.6',
        'transformers==2.11.0',
        'emoji==0.6.0',
        'recognizers-text-suite'
    ],
    extras_require=extras_requires,
)

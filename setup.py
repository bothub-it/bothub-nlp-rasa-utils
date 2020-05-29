from setuptools import setup, find_packages


setup(
    name='bothub_nlp_rasa_utils',
    version='0.1.1',
    description='Bothub NLP Rasa Utils',
    packages=find_packages(),
    install_requires=[
        'rasa==1.10.1',
    ],
)
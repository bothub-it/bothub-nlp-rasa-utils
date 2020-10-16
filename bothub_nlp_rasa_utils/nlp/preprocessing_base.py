import re
from .preprocessing_interface import PreprocessingInterface
from unidecode import unidecode
import emoji
# -*- coding: utf-8 -*-


def de_emojify(phrase):

    # emoji_pattern = re.compile("["
    #                            u"\U0001F600-\U0001F64F"  # emoticons
    #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                            u"\U0001F1F2-\U0001F1F4"  # Macau flag
    #                            u"\U0001F1E6-\U0001F1FF"  # flags
    #                            u"\U0001F600-\U0001F64F"
    #                            u"\U00002702-\U000027B0"
    #                            u"\U000024C2-\U0001F251"
    #                            u"\U0001f926-\U0001f937"
    #                            u"\U0001F1F2"
    #                            u"\U0001F1F4"
    #                            u"\U0001F620"
    #                            u"\u200d"
    #                            u"\u2640-\u2642"
    #                            u"\U0001F700-\U0001F77F"  # alchemical symbols
    #                            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    #                            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    #                            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    #                            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
    #                            u"\U00002702-\U000027B0"  # Dingbats
    #                            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    #                            "]+", flags=re.UNICODE)
    phrase = emoji.demojize(phrase)
    return phrase


class PreprocessingBase(PreprocessingInterface):

    def preprocess(self, phrase: str = None):
        # removing accent and lowercasing characters
        phrase = de_emojify(phrase)
        return unidecode(phrase.lower())

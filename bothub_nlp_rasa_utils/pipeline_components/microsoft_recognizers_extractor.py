import typing
from recognizers_suite import recognize_number, recognize_ordinal, recognize_age, recognize_currency, \
    recognize_dimension, recognize_temperature, recognize_datetime, recognize_phone_number, recognize_email
from recognizers_suite import Culture, ModelResult

from typing import Any, Dict, List, Text, Optional, Type
from rasa.nlu.constants import ENTITIES
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.training_data import Message

recognizers = {
    'number': recognize_number,
    'ordinal': recognize_ordinal,
    'age': recognize_age,
    'currency': recognize_currency,
    'dimension': recognize_dimension,
    'temperature': recognize_temperature,
    'datetime': recognize_datetime,
    'phone_number': recognize_phone_number,
    'email': recognize_email
}


def rasa_format(entity):
    return {
        'entity': entity.type_name,
        'start': entity.start,
        'end': entity.end,
        'value0': entity.text
    }


class MicrosoftRecognizersExtractor(EntityExtractor):
    defaults = {
        "dimensions": None
    }

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            language: Optional[Text] = None,
    ) -> None:
        super().__init__(component_config)
        self.language = language

    @classmethod
    def create(
            cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "MicrosoftRecognizersExtractor":

        return cls(component_config, config.language)

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER
        dimensions = self.component_config["dimensions"]
        print(self.language)
        print(dimensions)
        extracted = self.add_extractor_name(self.extract_entities(message.text, self.language, dimensions))
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)

    @staticmethod
    def extract_entities(user_input: str, culture: str, selected_dimensions):
        entities_group = []
        for dimension in recognizers:
            if dimension in selected_dimensions:
                entities = recognizers[dimension](user_input, culture)
                if entities:
                    for entity in entities:
                        entities_group.append(rasa_format(entity))

        print(entities_group)
        return entities_group

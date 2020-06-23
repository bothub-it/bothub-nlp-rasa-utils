from unidecode import unidecode
import re
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

from ..nlp.preprocessing_factory import PreprocessingFactory

class Preprocessing(Component):

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"language": None}

    def __init__(
        self, component_config: Optional[Dict[Text, Any]] = None, language: str = None
    ) -> None:
        super().__init__(component_config)
        self.language = component_config.get("language")


    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "Preprocessing":
        # component_config = override_defaults(cls.defaults, component_config)
        language = config.language
        return cls(component_config, language)

    def provide_context(self) -> Dict[Text, Any]:
        return {"language": self.language}

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component"""
        not_repeated_phrases = set()
        size = len(training_data.training_examples)
        subtract_idx = 0

        PREPROCESS_FACTORY = PreprocessFactory().get_preprocess(config.language)

        for idx in range(size):
            example_text = training_data.training_examples[idx - subtract_idx].text
            # removing accent and lowercasing characters
            example_text = unidecode(example_text.lower())

            PREPROCESS_FACTORY.preprocess(example_text)

            if example_text in not_repeated_phrases:
                # remove example at this index from training_examples
                training_data.training_examples.pop(idx - subtract_idx)
                subtract_idx += 1
            else:
                not_repeated_phrases.add(example_text)
                training_data.training_examples[idx - subtract_idx].text = example_text

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message."""
        APOSTROPHE_OPTIONS = ["'", "`"]

        # removing accent and lowercasing characters
        message.text = unidecode(message.text.lower())
        # remove apostrophe from the phrase (important be first than s_regex regex)
        for APOSTROPHE in APOSTROPHE_OPTIONS:
            message.text = message.text.replace(APOSTROPHE, "")
        
        PREPROCESS_FACTORY = PreprocessFactory().get_preprocess(config.language)

        PREPROCESS_FACTORY.preprocess(message.text)
import logging
import os
import re
from typing import Any, Dict, List, Optional, Text, Union

import rasa.utils.common
import rasa.utils.io

import rasa.nlu.utils.pattern_utils as pattern_utils

from rasa.nlu.model import Metadata
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractor
from ..nlp.preprocessing_base import PreprocessingBase

logger = logging.getLogger(__name__)


def _generate_lookup_regex(lookup_table: Dict[Text, Union[Text, List[Text]]]) -> Text:
    """Creates a regex pattern from the given lookup table.

    The lookup table is either a file or a list of entries.

    Args:
        lookup_table: The lookup table.

    Returns:
        The regex pattern.
    """
    lookup_elements = lookup_table["elements"]

    # if it's a list, it should be the elements directly
    if isinstance(lookup_elements, list):
        elements_to_regex = lookup_elements
    # otherwise it's a file path.
    else:
        elements_to_regex = pattern_utils.read_lookup_table_file(lookup_elements)

    # sanitize the regex, escape special characters
    preprocessor = PreprocessingBase()
    elements_sanitized = [re.escape(preprocessor.preprocess(e)) if not e.startswith('regex ')
                          else e.split('regex ')[1]
                          for e in elements_to_regex]

    # regex matching elements with word boundaries on either side
    return "(\\b" + "\\b|\\b".join(elements_sanitized) + "\\b)"


def _convert_lookup_tables_to_regex(
    training_data: TrainingData, use_only_entities: bool = False
) -> List[Dict[Text, Text]]:
    """Convert the lookup tables from the training data to regex patterns.
    Args:
        training_data: The training data.
        use_only_entities: If True only regex features with a name equal to a entity
          are considered.

    Returns:
        A list of regex patterns.
    """
    patterns = []
    for table in training_data.lookup_tables:
        if use_only_entities and table["name"] not in training_data.entities:
            continue
        regex_pattern = _generate_lookup_regex(table)
        # if file is empty
        if regex_pattern == r"(\b\b)":
            continue
        lookup_regex = {"name": table["name"], "pattern": regex_pattern}
        patterns.append(lookup_regex)

    return patterns


def extract_patterns(
    training_data: TrainingData,
    use_lookup_tables: bool = True,
    use_regexes: bool = True,
    use_only_entities: bool = False,
) -> List[Dict[Text, Text]]:
    """Extract a list of patterns from the training data.

    The patterns are constructed using the regex features and lookup tables defined
    in the training data.

    Args:
        training_data: The training data.
        use_only_entities: If True only lookup tables and regex features with a name
          equal to a entity are considered.
        use_regexes: Boolean indicating whether to use regex features or not.
        use_lookup_tables: Boolean indicating whether to use lookup tables or not.

    Returns:
        The list of regex patterns.
    """
    if not training_data.lookup_tables and not training_data.regex_features:
        return []

    patterns = []

    if use_regexes:
        patterns.extend(pattern_utils._collect_regex_features(training_data, use_only_entities))
    if use_lookup_tables:
        patterns.extend(
            _convert_lookup_tables_to_regex(training_data, use_only_entities)
        )

    return patterns


class RegexEntityExtractorCustom(EntityExtractor):
    """Searches for entities in the user's message using the lookup tables and regexes
    defined in the training data."""

    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        # use lookup tables to extract entities
        "use_lookup_tables": True,
        # use regexes to extract entities
        "use_regexes": True,
    }

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        super(RegexEntityExtractorCustom, self).__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        self.patterns = patterns or []

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        self.patterns = extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
            use_only_entities=False,
        )

        if not self.patterns:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RegexEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "RegexEntityExtractorCustom":

        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = rasa.shared.utils.io.read_json_file(regex_file)
            return RegexEntityExtractorCustom(meta, patterns=patterns)

        return RegexEntityExtractorCustom(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        rasa.shared.utils.io.dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}

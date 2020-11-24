import logging
import numpy as np
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.components import Component
from typing import Any, Optional, Text, List, Type, Dict, Tuple

from rasa.nlu.constants import NO_LENGTH_RESTRICTION

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTHS = {
    "bert_english": 512,
    "bert_portuguese": 512,
    "bert_multilang": 512,
    "bert": 512,
    "gpt": 512,
    "gpt2": 512,
    "xlnet": NO_LENGTH_RESTRICTION,
    "distilbert": 512,
    "roberta": 512,
}


class LanguageModelFeaturizerCustom(LanguageModelFeaturizer):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        skip_model_load: bool = False,
        hf_transformers_loaded: bool = False,
    ) -> None:
        """Initializes LanguageModelFeaturizer with the specified model.

        Args:
            component_config: Configuration for the component.
            skip_model_load: Skip loading the model for pytests.
            hf_transformers_loaded: Skip loading of model and metadata, use
            HFTransformers output instead.
        """
        super(LanguageModelFeaturizer, self).__init__(component_config)
        if hf_transformers_loaded:
            return
        self._load_model_metadata()
        self._load_model_instance(skip_model_load)

    def _load_model_metadata(self) -> None:

        from .registry import model_class_dict, model_weights_defaults

        self.model_name = self.component_config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create "
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self.component_config["model_weights"]
        self.cache_dir = self.component_config["cache_dir"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model weights: "
                f"{model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

        self.max_model_sequence_length = MAX_SEQUENCE_LENGTHS[self.model_name]

    def _load_model_instance(self, skip_model_load: bool) -> None:
        """Try loading the model instance.

        Args:
            skip_model_load: Skip loading the model instances to save time.
            This should be True only for pytests
        """
        if skip_model_load:
            # This should be True only during pytests
            return

        from .registry import (
            model_class_dict,
            model_weights_defaults,
            model_tokenizer_dict,
            from_pt_dict,
        )

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        try:
            from bothub_nlp_celery.app import nlp_language

            self.tokenizer, self.model = nlp_language
        except TypeError:
            logger.info(
                f"Model could not be retrieved from celery cache "
                f"Loading model {self.model_name} in memory"
            )
            self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
                model_weights_defaults[self.model_name], cache_dir=None
            )
            self.model = model_class_dict[self.model_name].from_pretrained(
                self.model_name,
                cache_dir=None,
                from_pt=from_pt_dict.get(self.model_name, False),
            )

        # Use a universal pad token since all transformer architectures do not have a
        # consistent token. Instead of pad_token_id we use unk_token_id because
        # pad_token_id is not set for all architectures. We can't add a new token as
        # well since vocabulary resizing is not yet supported for TF classes.
        # Also, this does not hurt the model predictions since we use an attention mask
        # while feeding input.
        self.pad_token_id = self.tokenizer.unk_token_id

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        """Clean up special chars added by tokenizers of language models.

        Many language models add a special char in front/back of (some) words. We clean
        up those chars as they are not
        needed once the features are already computed.

        Args:
            split_token_ids: List of token ids received as output from the language
            model specific tokenizer.
            token_strings: List of token strings received as output from the language
            model specific tokenizer.

        Returns: Cleaned up token ids and token strings.
        """
        from .registry import model_tokens_cleaners

        return model_tokens_cleaners[self.model_name](split_token_ids, token_strings)

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        """Add language model specific special tokens which were used during
        their training.

        Args:
            token_ids: List of token ids for each example in the batch.

        Returns: Augmented list of token ids for each example in the batch.
        """
        from .registry import model_special_tokens_pre_processors

        augmented_tokens = [
            model_special_tokens_pre_processors[self.model_name](example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sentence and sequence level representations for relevant tokens.

        Args:
            sequence_embeddings: Sequence level dense features received as output from
            language model.

        Returns: Sentence and sequence level representations.
        """
        from .registry import model_embeddings_post_processors

        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors[self.model_name](example_embedding)

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            np.array(post_processed_sequence_embeddings),
        )

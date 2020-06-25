import logging
from typing import Any, Dict, List, Text, Tuple, Optional

from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token
import rasa.utils.train_utils as train_utils
import numpy as np

from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from bothub_nlp_celery.app import nlp_language, nlp_tokenizer
from bothub_nlp_celery.app import nlp_bert


from rasa.nlu.constants import (
    TEXT,
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    TOKEN_IDS,
    TOKENS,
    SENTENCE_FEATURES,
    SEQUENCE_FEATURES,
)

logger = logging.getLogger(__name__)


from bothub_nlp_celery import settings


# nlp_language, nlp_tokenizer = nlp_bert()

class HFTransformersNLPCustom(HFTransformersNLP):
    """Utility Component for interfacing between Transformers library and Rasa OS.

    The transformers(https://github.com/huggingface/transformers) library
    is used to load pre-trained language models like BERT, GPT-2, etc.
    The component also tokenizes and featurizes dense featurizable attributes of each
    message.
    """

    defaults = {
        # name of the language model to load.
        "model_name": "bert",
        # Pre-Trained weights to be loaded(string)
        "model_weights": None,
        # an optional path to a specific directory to download
        # and cache the pre-trained model weights.
        "cache_dir": None,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(HFTransformersNLP, self).__init__(component_config)

        self._load_model()
        self.whitespace_tokenizer = WhitespaceTokenizer()

    def _load_model(self) -> None:
        """Try loading the model"""

        from .registry import (
            model_class_dict,
            model_weights_defaults,
            model_tokenizer_dict,
            from_pt_dict,
        )

        self.model_name = self.component_config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))}or create"
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
            

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        # self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
        #     self.model_weights, cache_dir=self.cache_dir
        # )
        # self.model = model_class_dict[self.model_name].from_pretrained(
        #     self.model_weights, cache_dir=self.cache_dir, from_pt=from_pt_dict.get(self.model_name, False)
        # )

        # self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
        #     model_weights_defaults[self.model_name], cache_dir=None
        # )
        #
        # self.model = model_class_dict[self.model_name].from_pretrained(
        #     model_weights_defaults[self.model_name], cache_dir=None,
        #     from_pt=from_pt_dict.get(self.model_name, False)
        # )
        from pprint import pprint

        self.tokenizer = nlp_tokenizer
        self.model = nlp_language


        # Use a universal pad token since all transformer architectures do not have a
        # consistent token. Instead of pad_token_id we use unk_token_id because
        # pad_token_id is not set for all architectures. We can't add a new token as
        # well since vocabulary resizing is not yet supported for TF classes.
        # Also, this does not hurt the model predictions since we use an attention mask
        # while feeding input.
        self.pad_token_id = self.tokenizer.unk_token_id
        logger.debug(f"Loaded Tokenizer and Model for {self.model_name}")

    # def train(
    #         self,
    #         training_data: TrainingData,
    #         config: Optional[RasaNLUModelConfig] = None,
    #         **kwargs: Any,
    # ) -> None:
    #     """Compute tokens and dense features for each message in training data.
    #     Args:
    #         training_data: NLU training data to be tokenized and featurized
    #         config: NLU pipeline config consisting of all components.
    #     """
    #
    #     batch_size = 64
    #
    #     for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
    #         non_empty_examples = list(
    #             filter(lambda x: x.get(attribute), training_data.training_examples)
    #         )
    #         batch_start_index = 0
    #         while batch_start_index < len(non_empty_examples):
    #             batch_end_index = min(
    #                 batch_start_index + batch_size, len(non_empty_examples)
    #             )
    #             # Collect batch examples
    #             batch_messages = non_empty_examples[batch_start_index:batch_end_index]
    #             # Construct a doc with relevant features
    #             # extracted(tokens, dense_features)
    #             batch_docs = self._get_docs_for_batch(batch_messages, attribute)
    #             for index, ex in enumerate(batch_messages):
    #                 ex.set(LANGUAGE_MODEL_DOCS[attribute], batch_docs[index])
    #             batch_start_index += batch_size
    #
    # def _get_docs_for_batch(
    #         self, batch_examples: List[Message], attribute: Text
    # ) -> List[Dict[Text, Any]]:
    #     """Compute language model docs for all examples in the batch.
    #     Args:
    #         batch_examples: Batch of message objects for which language model docs need to be computed.
    #         attribute: Property of message to be processed, one of ``TEXT`` or ``RESPONSE``.
    #     Returns:
    #         List of language model docs for each message in batch.
    #     """
    #
    #     batch_tokens, batch_token_ids = self._get_token_ids_for_batch(
    #         batch_examples, attribute
    #     )
    #     (
    #         batch_sentence_features,
    #         batch_sequence_features,
    #     ) = self._get_model_features_for_batch(batch_token_ids)
    #     # A doc consists of
    #     # {'token_ids': ..., 'tokens': ..., 'sequence_features': ..., 'sentence_features': ...}
    #     batch_docs = []
    #     for index in range(len(batch_examples)):
    #         doc = {
    #             TOKEN_IDS: batch_token_ids[index],
    #             TOKENS: batch_tokens[index],
    #             SEQUENCE_FEATURES: batch_sequence_features[index],
    #             SENTENCE_FEATURES: np.reshape(batch_sentence_features[index], (1, -1)),
    #         }
    #         batch_docs.append(doc)
    #     return batch_docs
    #
    # def _get_model_features_for_batch(
    #     self, batch_token_ids: List[List[int]]
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Compute dense features of each example in the batch.
    #     We first add the special tokens corresponding to each language model. Next, we add appropriate padding
    #     and compute a mask for that padding so that it doesn't affect the feature computation. The padded batch is next
    #     fed to the language model and token level embeddings are computed. Using the pre-computed mask, embeddings for
    #     non-padding tokens are extracted and subsequently sentence level embeddings are computed.
    #     Args:
    #         batch_token_ids: List of token ids of each example in the batch.
    #     Returns:
    #         Sentence and token level dense representations.
    #     """
    #     # Let's first add tokenizer specific special tokens to all examples
    #     batch_token_ids_augmented = self._add_lm_specific_special_tokens(
    #         batch_token_ids
    #     )
    #     # Let's first add padding so that whole batch can be fed to the model
    #     actual_sequence_lengths, padded_token_ids = self._add_padding_to_batch(
    #         batch_token_ids_augmented
    #     )
    #     # Compute attention mask based on actual_sequence_length
    #     batch_attention_mask = self._compute_attention_mask(actual_sequence_lengths)
    #     # Get token level features from the model
    #     sequence_hidden_states = self._compute_batch_sequence_features(
    #         batch_attention_mask, padded_token_ids
    #     )
    #     # Extract features for only non-padding tokens
    #     sequence_nonpadded_embeddings = self._extract_nonpadded_embeddings(
    #         sequence_hidden_states, actual_sequence_lengths
    #     )
    #     # Extract sentence level and post-processed features
    #     (
    #         sentence_embeddings,
    #         sequence_final_embeddings,
    #     ) = self._post_process_sequence_embeddings(sequence_nonpadded_embeddings)
    #     return sentence_embeddings, sequence_final_embeddings
    #
    # def _compute_batch_sequence_features(
    #         self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    # ) -> np.ndarray:
    #     """Feed the padded batch to the language model.
    #     Args:
    #         batch_attention_mask: Mask of 0s and 1s which indicate whether the token is a padding token or not.
    #         padded_token_ids: Batch of token ids for each example. The batch is padded and hence can be fed at once.
    #     Returns:
    #         Sequence level representations from the language model.
    #     """
    #     model_outputs = self.model(
    #         np.array(padded_token_ids), attention_mask=np.array(batch_attention_mask)
    #     )
    #     # sequence hidden states is always the first output from all models
    #     sequence_hidden_states = model_outputs[0]
    #
    #     sequence_hidden_states = sequence_hidden_states.numpy()
    #     return sequence_hidden_states
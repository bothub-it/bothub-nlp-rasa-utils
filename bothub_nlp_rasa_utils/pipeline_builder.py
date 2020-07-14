from rasa.nlu.config import RasaNLUModelConfig
from bothub_nlp_celery import settings


def add_spacy_nlp():
    return {"name": "bothub_nlp_rasa_utils.pipeline_components.spacy_nlp.SpacyNLP"}


def add_whitespace_tokenizer():
    return {"name": "WhitespaceTokenizer"}


def add_preprocessing(update):
    return {
        "name": "bothub_nlp_rasa_utils.pipeline_components.preprocessing.Preprocessing",
        "language": update.get("language"),
    }


def add_countvectors_featurizer(update):
    if update.get("use_analyze_char"):
        return {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char",
            "min_ngram": 3,
            "max_ngram": 3,
            "token_pattern": "(?u)\\b\\w+\\b",
        }

    else:
        return {"name": "CountVectorsFeaturizer", "token_pattern": "(?u)\\b\\w+\\b"}


def add_embedding_intent_classifier():
    return {
        "name": "bothub_nlp_rasa_utils.pipeline_components.diet_classifier.DIETClassifierCustom",
        "hidden_layers_sizes": {"text": [256, 128]},
        "number_of_transformer_layers": 0,
        "weight_sparsity": 0,
        "intent_classification": True,
        "entity_recognition": True,
        "use_masked_language_model": False,
        "BILOU_flag": False,
    }


def add_diet_classifier():
    return {"name": "bothub_nlp_rasa_utils.pipeline_components.diet_classifier.DIETClassifierCustom", "entity_recognition": True, "BILOU_flag": False}


def legacy_internal_config(update):
    pipeline = [
        add_whitespace_tokenizer(),  # Tokenizer
        add_countvectors_featurizer(update),  # Featurizer
        add_embedding_intent_classifier(),  # Intent Classifier
    ]
    return pipeline


def legacy_external_config(update):
    pipeline = [
        {"name": "SpacyTokenizer"},  # Tokenizer
        {"name": "SpacyFeaturizer"},  # Spacy Featurizer
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_embedding_intent_classifier(),  # intent classifier
    ]
    return pipeline


def transformer_network_diet_config(update):
    pipeline = [
        add_whitespace_tokenizer(),
        add_countvectors_featurizer(update),  # Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]
    return pipeline


def transformer_network_diet_word_embedding_config(update):
    pipeline = [
        {"name": "SpacyTokenizer"},  # Tokenizer
        {"name": "SpacyFeaturizer"},  # Spacy Featurizer
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]
    return pipeline


def transformer_network_diet_bert_config(update):
    pipeline = [
        {  # NLP
            "name": "bothub_nlp_rasa_utils.pipeline_components.hf_transformer.HFTransformersNLPCustom",
            "model_name": "bert_portuguese",
        },
        {  # Tokenizer
            "name": "bothub_nlp_rasa_utils.pipeline_components.lm_tokenizer.LanguageModelTokenizerCustom",
            "intent_tokenization_flag": False,
            "intent_split_symbol": "_",
        },
        {  # Bert Featurizer
            "name": "bothub_nlp_rasa_utils.pipeline_components.lm_featurizer.LanguageModelFeaturizerCustom"
        },
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]
    return pipeline


def get_rasa_nlu_config_from_update(update):  # pragma: no cover
    spacy_algorithms = [
        "neural_network_external",
        "transformer_network_diet_word_embedding",
    ]
    pipeline = [add_preprocessing(update)]
    if update.get("use_name_entities") or update.get("algorithm") in spacy_algorithms:
        pipeline.append(add_spacy_nlp())

    if update.get("algorithm") == "neural_network_internal":
        pipeline.extend(legacy_internal_config(update))
    elif update.get("algorithm") == "neural_network_external":
        pipeline.extend(legacy_external_config(update))
    elif update.get("algorithm") == "transformer_network_diet":
        pipeline.extend(transformer_network_diet_config(update))
    elif update.get("algorithm") == "transformer_network_diet_word_embedding":
        pipeline.extend(transformer_network_diet_word_embedding_config(update))
    elif update.get("algorithm") == "transformer_network_diet_bert":
        pipeline.extend(transformer_network_diet_bert_config(update))
    else:
        return

    # spacy named entity recognition
    if update.get("use_name_entities"):
        pipeline.append({"name": "SpacyEntityExtractor"})

    return RasaNLUModelConfig(
        {"language": update.get("language"), "pipeline": pipeline}
    )

from rasa.nlu.config import RasaNLUModelConfig
from bothub_nlp_celery.utils import choose_best_algorithm, ALGORITHM_TO_LANGUAGE_MODEL
from bothub_nlp_celery import settings
from .pipeline_components.registry import language_to_model


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
            "model_name": language_to_model.get(update.get("language")),
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


def get_rasa_nlu_config(update):

    pipeline = []

    # algorithm = choose_best_algorithm(update.get("language"))
    algorithm = update.get('algorithm')
    language = update.get('language')
    
    model = ALGORITHM_TO_LANGUAGE_MODEL[algorithm]
    if (model == 'SPACY' and language not in settings.SPACY_LANGUAGES) or (
            model == 'BERT' and language not in settings.BERT_LANGUAGES):
        algorithm = "transformer_network_diet"

    print('languague:', language)
    print('algorithm:', algorithm)
    
    pipeline.append(add_preprocessing(update))
    if algorithm == "neural_network_internal":
        pipeline.extend(legacy_internal_config(update))
    elif algorithm == "neural_network_external":
        pipeline.append(add_spacy_nlp())
        pipeline.extend(legacy_external_config(update))
        if update.get("use_name_entities"):
            pipeline.append({"name": "SpacyEntityExtractor"})
    elif algorithm == "transformer_network_diet_bert":
        pipeline.extend(transformer_network_diet_bert_config(update))
    elif algorithm == "transformer_network_diet_word_embedding":
        pipeline.append(add_spacy_nlp())
        pipeline.extend(transformer_network_diet_word_embedding_config(update))
        if update.get("use_name_entities"):
            pipeline.append({"name": "SpacyEntityExtractor"})
    else:
        pipeline.extend(transformer_network_diet_config(update))

    print(f"New pipeline: {pipeline}")

    return RasaNLUModelConfig(
        {
            "language": update.get("language"),
            "pipeline": pipeline
        }
    )

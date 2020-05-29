from rasa.nlu.config import RasaNLUModelConfig

def add_whitespace_tokenizer():
    return {"name": "WhitespaceTokenizer"}


def add_preprocessing(update):
    return {
        "name": "pipeline_components.preprocessing.Preprocessing",
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
        "name": "DIETClassifier",
        "hidden_layers_sizes": {"text": [256, 128]},
        "number_of_transformer_layers": 0,
        "weight_sparsity": 0,
        "intent_classification": True,
        "entity_recognition": False,
        "use_masked_language_model": False,
        "BILOU_flag": False,
    }


def add_diet_classifier():
    return {"name": "DIETClassifier", "entity_recognition": False, "BILOU_flag": False}


def legacy_internal_config(update):
    pipeline = [
        add_whitespace_tokenizer(),  # Tokenizer
        add_countvectors_featurizer(update),  # Featurizer
        add_embedding_intent_classifier(),  # Intent Classifier
    ]

    return pipeline


def legacy_external_config(update):
    pipeline = [
        {"name": "SpacyNLP"},  # Language Model
        {"name": "SpacyTokenizer"},  # Tokenizer
        {"name": "SpacyFeaturizer"},  # Spacy Featurizer
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_embedding_intent_classifier(),  # intent classifier
    ]

    return pipeline


def transformer_network_diet_config(update):
    pipeline = [
        add_preprocessing(update),  # Preprocessing
        add_whitespace_tokenizer(),  # Tokenizer
        add_countvectors_featurizer(update),  # Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]

    return pipeline


def transformer_network_diet_word_embedding_config(update):
    pipeline = [
        add_preprocessing(update),  # Preprocessing
        {"name": "SpacyNLP"},  # Language Model
        {"name": "SpacyTokenizer"},  # Tokenizer
        {"name": "SpacyFeaturizer"},  # Spacy Featurizer
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]

    return pipeline


def transformer_network_diet_bert_config(update):
    pipeline = [
        add_preprocessing(update),
        {  # NLP
            "name": "pipeline_components.HFTransformerNLP.HFTransformersNLP",
            "model_name": "bert_portuguese",
        },
        {  # Tokenizer
            "name": "pipeline_components.lm_tokenizer.LanguageModelTokenizerCustom",
            "intent_tokenization_flag": False,
            "intent_split_symbol": "_",
        },
        {  # Bert Featurizer
            "name": "pipeline_components.lm_featurizer.LanguageModelFeaturizerCustom"
        },
        add_countvectors_featurizer(update),  # Bag of Words Featurizer
        add_diet_classifier(),  # Intent Classifier
    ]

    return pipeline


def get_rasa_nlu_config_from_update(update):  # pragma: no cover
    if update.get("algorithm") == "neural_network_internal":
        pipeline = legacy_internal_config(update)
    elif update.get("algorithm") == "neural_network_external":
        pipeline = legacy_external_config(update)
    elif update.get("algorithm") == "transformer_network_diet":
        pipeline = transformer_network_diet_config(update)
    elif update.get("algorithm") == "transformer_network_diet_word_embedding":
        pipeline = transformer_network_diet_word_embedding_config(update)
    elif update.get("algorithm") == "transformer_network_diet_bert":
        pipeline = transformer_network_diet_bert_config(update)
    else:
        return

    # entity extractor
    pipeline.append({"name": "CRFEntityExtractor"})

    # spacy named entity recognition
    if update.get("use_name_entities"):
        pipeline.append({"name": "SpacyEntityExtractor"})

    return RasaNLUModelConfig(
        {"language": update.get("language"), "pipeline": pipeline}
    )


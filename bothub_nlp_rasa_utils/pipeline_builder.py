from rasa.nlu.config import RasaNLUModelConfig


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


def add_diet_classifier():
    return {"name": "bothub_nlp_rasa_utils.pipeline_components.diet_classifier.DIETClassifierCustom", "entity_recognition": True, "BILOU_flag": False}


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


def get_algorithm_info():
    # todo: get data from config file / populate languages

    # Sorted by priority
    # last element -> default algorithm
    return [
        {
            "name": "transformer_network_diet_bert",
            "supported_languages": [],
            "config": transformer_network_diet_bert_config
        },
        {
            "name": "transformer_network_diet_word_embedding",
            "supported_languages": [],
            "config": transformer_network_diet_word_embedding_config
        },
        {
            "name": "transformer_network_diet",
            "supported_languages": ["all"],
            "config": transformer_network_diet_config
        }
    ]


def choose_best_algorithm(update):

    supported_algorithms = get_algorithm_info()

    for model in supported_algorithms[:-1]:
        if update.get("language") in model["supported_languages"]:
            return model

    # default algorithm
    return supported_algorithms[len(supported_algorithms)-1]


def get_rasa_nlu_config(update):

    pipeline = []

    chosen_model = choose_best_algorithm(update)

    pipeline.append(add_preprocessing(update))

    if update.get("use_name_entities") or chosen_model["name"] == "transformer_network_diet_word_embedding":
        pipeline.append(add_spacy_nlp())

    pipeline.extend(chosen_model["config"](update))

    if update.get("use_name_entities"):
        pipeline.append({"name": "SpacyEntityExtractor"})

    print(f"New pipeline: {pipeline}")

    return RasaNLUModelConfig(
        {
            "language": update.get("language"),
            "pipeline": pipeline
        }
    )
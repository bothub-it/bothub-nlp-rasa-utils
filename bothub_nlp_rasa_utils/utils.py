import io
import logging
import bothub_backend
import contextvars
import argparse
from tempfile import mkdtemp
from decouple import config
from rasa.nlu import components
from rasa.nlu.config import RasaNLUModelConfig

from rasa.nlu.model import Interpreter
from .persistor import BothubPersistor


def null_result(text):
    return {
        'intent':
            {
                'name': None,
                'confidence': 0.0
            },
        'entities': [],
        'intent_ranking': [],
        'text': text
    }


def is_text_valid(text):
    if text.replace(" ", ""):
        return True
    return False


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def backend():
    PARSER = argparse.ArgumentParser()

    # Input Arguments
    PARSER.add_argument(
        '--base_url',
        help='Base URL API Engine.',
        type=str, default=None)

    ARGUMENTS, _ = PARSER.parse_known_args()

    return bothub_backend.get_backend(
        "bothub_backend.bothub.BothubBackend",
        ARGUMENTS.base_url if ARGUMENTS.base_url else config("BOTHUB_ENGINE_URL", default="https://api.bothub.it"),
    )


def get_examples_request(update_id, repository_authorization):  # pragma: no cover
    start_examples = backend().request_backend_get_examples(
        update_id, False, None, repository_authorization
    )

    examples = start_examples.get("results")

    page = start_examples.get("next")

    if page:
        while True:
            request_examples_page = backend().request_backend_get_examples(
                update_id, True, page, repository_authorization
            )

            examples += request_examples_page.get("results")

            if request_examples_page.get("next") is None:
                break

            page = request_examples_page.get("next")

    return examples


class UpdateInterpreters:
    interpreters = {}

    def get(
        self, repository_version, repository_authorization, rasa_version, use_cache=True
    ):
        update_request = backend().request_backend_parse_nlu_persistor(
            repository_version, repository_authorization, rasa_version
        )

        repository_name = (
            f"{update_request.get('version_id')}_"
            f"{update_request.get('total_training_end')}_"
            f"{update_request.get('language')}"
        )

        interpreter = self.interpreters.get(repository_name)

        if interpreter and use_cache:
            return interpreter
        persistor = BothubPersistor(
            repository_version, repository_authorization, rasa_version
        )
        model_directory = mkdtemp()
        persistor.retrieve(str(update_request.get("repository_uuid")), model_directory)
        self.interpreters[repository_name] = Interpreter(
            None, {"language": update_request.get("language")}
        ).load(model_directory, components.ComponentBuilder(use_cache=False))
        return self.get(repository_version, repository_authorization, rasa_version)



class PokeLoggingHandler(logging.StreamHandler):
    def __init__(self, pl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pl = pl

    def emit(self, record):
        if self.pl.cxt.get(default=None) is self.pl:
            super().emit(record)


class PokeLogging:
    def __init__(self, loggingLevel=logging.DEBUG):
        self.loggingLevel = loggingLevel

    def __enter__(self):
        self.cxt = contextvars.ContextVar(self.__class__.__name__)
        self.cxt.set(self)
        logging.captureWarnings(True)
        self.logger = logging.getLogger()
        self.logger.setLevel(self.loggingLevel)
        self.stream = io.StringIO()
        self.handler = PokeLoggingHandler(self, self.stream)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.handler.setLevel(self.loggingLevel)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        return self.stream

    def __exit__(self, *args):
        self.logger.removeHandler(self.logger)


update_interpreters = UpdateInterpreters()

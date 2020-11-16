from collections import OrderedDict
from rasa.nlu import __version__ as rasa_version
from .utils import update_interpreters
from .utils import update_interpreters, is_text_valid, null_result


def get_interpreter(
    repository_version,
    repository_authorization,
    use_cache,
):
    interpreter = update_interpreters.get(
        repository_version, repository_authorization, rasa_version, use_cache=use_cache
    )
    return interpreter


def parse_interpreter(interpreter, text):
    if is_text_valid(text):
        return interpreter.parse(text)
    else:
        return null_result(text)

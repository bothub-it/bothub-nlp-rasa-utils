import bothub_backend

def backend():
    return bothub_backend.get_backend(
        "bothub_backend.bothub.BothubBackend",
        config("BOTHUB_ENGINE_URL", default="https://api.bothub.it"),
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

    def get(self, repository_version, repository_authorization, use_cache=True):
        update_request = backend().request_backend_parse_nlu(
            repository_version, repository_authorization
        )

        repository_name = (
            f"{update_request.get('version_id')}_"
            f"{update_request.get('total_training_end')}_"
            f"{update_request.get('language')}"
        )

        interpreter = self.interpreters.get(repository_name)

        if interpreter and use_cache:
            return interpreter
        persistor = BothubPersistor(repository_version, repository_authorization)
        model_directory = mkdtemp()
        persistor.retrieve(str(update_request.get("repository_uuid")), model_directory)
        self.interpreters[repository_name] = Interpreter(
            None, {"language": update_request.get("language")}
        ).load(model_directory, components.ComponentBuilder(use_cache=False))
        return self.get(repository_version, repository_authorization)


update_interpreters = UpdateInterpreters()

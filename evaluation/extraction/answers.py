"""Extract final answers from the common result contract."""


def extract(records: list[dict]) -> list[dict]:
    return [
        {key: record[key] for key in ("question_id", "answer", "execution_error")}
        for record in records
    ]

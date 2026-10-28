"""Extract submitted trajectories and cumulative execution cost traces."""


def extract(records: list[dict]) -> list[dict]:
    return [
        {
            "question_index": str(record["question_id"]),
            "tool_calls": record["tool_calls"],
            "cumulative_tool_calls": record["cumulative_tool_calls"],
            "output_roots": record["output_roots"],
        }
        for record in records
    ]

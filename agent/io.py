"""I/O helpers for benchmark questions, run results, and chat logging."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Tuple, Union

PathLike = Union[str, Path]


def save_chat_message(
    chat_log_path: PathLike,
    role: str,
    content: Any,
    question_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(chat_log_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = {**(metadata or {}), "question_id": question_id}
    name = (
        "user" if role == "user" else question_id if role == "assistant" else "system"
    )

    record = {
        "id": uuid.uuid4().hex,
        "name": name,
        "role": role,
        "content": content,
        "metadata": md,
        "timestamp": ts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _JsonFormatter(logging.Formatter):
    """Format a log record using the structured conversation log schema."""

    def format(self, record: logging.LogRecord) -> str:
        args = record.args if record.args else ()
        log_record = {
            "question_index": args[0] if len(args) > 0 else "unknown",
            "timestamp": self.formatTime(record, self.datefmt),
            "conversations": args[1] if len(args) > 1 else [],
            "final_answer": args[2] if len(args) > 2 else None,
        }
        return json.dumps(log_record, ensure_ascii=False, indent=4)


def create_experiment_logger(
    output_dir: PathLike,
    model_name: str,
    *,
    autoplanning: bool = True,
) -> logging.Logger:
    """Create a per-experiment file logger that writes JSON records.

    Produces ``{model_name}_{AP|IF}_langchain.log`` inside *output_dir*, one
    JSON object per line, for the current runner.
    """
    mode_tag = "AP" if autoplanning else "IF"
    log_path = Path(output_dir) / f"{model_name}_{mode_tag}_langchain.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"agent.experiment.{log_path.stem}.{id(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)

    return logger


def _filter_chat_through_attempt(chat_path: Path, n: int) -> list[str]:
    """Keep attempts 1..n independently for each question in interleaved logs."""
    counts: dict[str, int] = {}
    out = []
    for raw in chat_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        message = json.loads(raw)
        qid = str(message["metadata"]["question_id"])
        if message["role"] == "user":
            counts[qid] = counts.get(qid, 0) + 1
        if 0 < counts.get(qid, 0) <= n:
            out.append(raw)
    return out


def _iter_json_log_blocks(log_path: Path) -> Iterable[Tuple[str, Any]]:
    """Yield ``(block_text, parsed_record_or_None)`` for each JSON value in a
    ``.log`` file.

    Earlier formatters wrote each record with ``indent=4`` and
    handlers append them back-to-back separated by newlines, so a ``.log``
    file is a stream of pretty-printed JSON values (not JSONL).

    Splitting by raw ``{``/``}`` counts is unsafe because tool-output /
    assistant-content strings frequently contain unbalanced braces which
    silently glue multiple records into one huge (unparseable) block —
    downstream that causes ``seed_resume_files`` to keep only the first
    couple of records of a 200+ record run.

    Instead, delegate the split to ``json.JSONDecoder.raw_decode``, which
    consumes exactly one complete JSON value from a string and returns the
    index where it stopped. That correctly handles ``{``/``}`` inside string
    literals because it *is* the same parser used everywhere else.

    ``block_text`` is the exact substring of the source file that produced
    the JSON value (leading whitespace stripped). ``parsed_record`` is the
    decoded object, or ``None`` if the fragment could not be decoded (kept
    for symmetry with the previous behavior of preserving unparseable
    trailers).
    """
    text = log_path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, idx=i)
        except json.JSONDecodeError:
            # Preserve whatever junk remains as one final unparseable block so
            # callers can still inspect it (matching the earlier append behavior for
            # blocks that fail ``json.loads``). Then stop — we can't safely
            # skip past malformed JSON.
            tail = text[i:].rstrip()
            if tail:
                yield tail, None
            return
        block_text = text[i:end]
        yield block_text, obj
        i = end


def _filter_log_through_attempt(log_path: Path, n: int) -> list[str]:
    """Return the first *n* occurrences of each ``question_index`` in a ``.log``.

    The ``.log`` is a sequence of JSON records (possibly multi-line with
    ``indent=4``).  Each record has a ``question_index`` key.
    """
    counts: dict[str, int] = {}
    out: list[str] = []

    for block, record in _iter_json_log_blocks(log_path):
        if not isinstance(record, dict):
            out.append(block)
            continue
        qidx = str(record.get("question_index", ""))
        if not qidx:
            out.append(block)
            continue
        counts[qidx] = counts.get(qidx, 0) + 1
        if counts[qidx] <= n:
            out.append(block)

    return out


def seed_resume_files(
    resume_dir: PathLike, experiment_dir: PathLike, through_attempt: int = 1
) -> None:
    """Copy completed result rounds, earlier plugin artifacts and filtered logs.

    Evaluation is derived from these raw results inside the destination run.
    """
    import shutil

    src = Path(resume_dir)
    dst = Path(experiment_dir)
    dst.mkdir(parents=True, exist_ok=True)

    (dst / "traces").mkdir(parents=True, exist_ok=True)
    for k in range(1, through_attempt + 1):
        jsonl = src / "traces" / f"attempt_{k}.jsonl"
        dst_jsonl = dst / "traces" / f"attempt_{k}.jsonl"
        if jsonl.exists() and not dst_jsonl.exists():
            shutil.copy2(jsonl, dst_jsonl)

    for k in range(1, through_attempt):
        for source in (src / "plugins").glob(f"*/attempt_{k}"):
            shutil.copytree(source, dst / source.relative_to(src), dirs_exist_ok=True)

    chat_files = sorted(src.glob("*.chat"))
    dst_chat = dst / "run.chat"
    if chat_files and not dst_chat.exists():
        filtered = _filter_chat_through_attempt(chat_files[0], through_attempt)
        dst_chat.write_text(
            "\n".join(filtered) + "\n" if filtered else "", encoding="utf-8"
        )

    for log_file in sorted(src.glob("*_langchain.log")):
        dst_log = dst / log_file.name
        if not dst_log.exists():
            filtered = _filter_log_through_attempt(log_file, through_attempt)
            dst_log.write_text(
                "\n".join(filtered) + "\n" if filtered else "", encoding="utf-8"
            )


def cleanup_chat_after_continue(chat_path: PathLike, error_qids: set[str]) -> None:
    """Remove only unfinished/error questions, regardless of message interleaving."""
    path = Path(chat_path)
    if not path.exists():
        return
    kept = [
        raw
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
        and str(json.loads(raw)["metadata"]["question_id"]) not in error_qids
    ]
    path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")


def cleanup_log_after_continue(log_path: PathLike, error_qids: set[str]) -> None:
    """Remove unfinished/error question records from the structured log."""
    path = Path(log_path)
    if not path.exists():
        return
    kept = [
        block
        for block, record in _iter_json_log_blocks(path)
        if not isinstance(record, dict)
        or str(record.get("question_index", "")) not in error_qids
    ]
    path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")


def _is_error_result(record: MutableMapping[str, Any]) -> bool:
    return record["execution_error"]


def load_completed_results(
    jsonl_path: PathLike,
) -> Tuple[List[MutableMapping[str, Any]], List[str]]:
    """Load a partial ``traces/attempt_N.jsonl`` and separate completed
    results from error results.

    Returns ``(completed_results, error_question_ids)`` where
    *completed_results* preserves the original order and *error_question_ids*
    lists question IDs whose answers are API/runtime errors.
    """
    path = Path(jsonl_path)
    completed: List[MutableMapping[str, Any]] = []
    error_ids: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if _is_error_result(r):
                error_ids.append(str(r["question_id"]))
            else:
                completed.append(r)

    return completed, error_ids


def load_attempts_through(
    resume_dir: PathLike, n: int
) -> Tuple[List[dict], List[str], Path]:
    """Load attempts 1..*n* from *resume_dir*.

    Returns ``(loaded_attempts, failed_ids_of_attempt_n, source_chat_path)``
    where each element of *loaded_attempts* is a dict with keys
    ``attempt``, ``results``, ``failed_ids``, ``token_usage``.

    The returned *failed_ids* are from **attempt n** (not attempt 1), which
    drives the next attempt.
    """
    base = Path(resume_dir)
    from agent.tokens import TOKEN_FIELDS, sum_usage_ledger

    loaded_attempts: List[dict] = []
    expected_ids = set(
        json.loads((base / "dataset_snapshot.json").read_text(encoding="utf-8"))[
            "question_ids"
        ]
    )

    for k in range(1, n + 1):
        attempt_file = base / "traces" / f"attempt_{k}.jsonl"
        if not attempt_file.exists():
            raise FileNotFoundError(f"traces/attempt_{k}.jsonl not found in {base}")

        results: List[MutableMapping[str, Any]] = []
        with open(attempt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        actual_ids = [str(result["question_id"]) for result in results]
        if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
            raise ValueError(
                f"Attempt {k} is incomplete or contains duplicate/unexpected question IDs"
            )
        if any(result["attempt"] != k for result in results):
            raise ValueError(f"Attempt number mismatch in {attempt_file}")
        failed_ids_k = [
            str(result["question_id"])
            for result in results
            if result["heuristic_issue"]
        ]
        expected_ids = set(failed_ids_k)

        loaded_attempts.append(
            {
                "attempt": k,
                "results": results,
                "failed_ids": failed_ids_k,
                "token_usage": {
                    key: sum(result["token_usage"][key] for result in results)
                    for key in TOKEN_FIELDS
                },
                "plugin_token_usage": dict.fromkeys(TOKEN_FIELDS, 0),
            }
        )
        # Feedback after n is generated anew by the resumed experiment.
        if k < n:
            for ledger in (base / "plugins").glob(f"*/attempt_{k}/model_usage.jsonl"):
                usage = sum_usage_ledger(ledger)
                for key in TOKEN_FIELDS:
                    loaded_attempts[-1]["plugin_token_usage"][key] += usage[key]

    final_failed_ids = loaded_attempts[-1]["failed_ids"]

    chat_files = sorted(base.glob("*.chat"))
    if not chat_files:
        raise FileNotFoundError(f"Chat log file not found in {base}")

    return loaded_attempts, final_failed_ids, chat_files[0]

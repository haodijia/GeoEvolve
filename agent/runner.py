"""Dataset-independent experiment execution with composable agent components."""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any

from agent.agent import create_agent, handle_question, load_config
from agent.answers import (
    aggregate_results,
    compute_statistics,
    heuristic_critic_has_issue,
)
from agent.config import ExperimentConfig
from agent.io import (
    cleanup_chat_after_continue,
    cleanup_log_after_continue,
    create_experiment_logger,
    load_attempts_through,
    load_completed_results,
    seed_resume_files,
)
from agent.tokens import TOKEN_FIELDS
from benchmark.dataset import Dataset
from plugins import create_plugins
from plugins.base import AgentPlugin, PluginContext


def _write_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _flush_results(path: Path, results: list[dict]) -> None:
    partial = path.with_suffix(".jsonl.tmp")
    partial.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    partial.replace(path)


def clear_tool_outputs(directory: Path) -> None:
    """Empty the experiment's scratch directory without following symlinks."""
    if directory.is_symlink():
        raise ValueError(f"Tool output directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for artifact in directory.iterdir():
        if artifact.is_dir() and not artifact.is_symlink():
            shutil.rmtree(artifact)
        else:
            artifact.unlink()


async def run_single_attempt(
    agent: Any,
    questions: list[dict],
    attempt: int,
    chat_log_path: Path,
    config: ExperimentConfig,
    plugins: list[AgentPlugin],
    output_dir: Path,
    logger: Any,
    completed_results: list[dict],
) -> dict:
    results = []
    completed = {str(item["question_id"]): item for item in completed_results}
    path = output_dir / "traces" / f"attempt_{attempt}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tool_output_dir = output_dir / "out"
    clear_tool_outputs(tool_output_dir)
    for question in questions:
        qid = str(question["question_id"])
        if qid in completed:
            results.append(completed[qid])
            continue
        try:
            answer, usage, calls = await handle_question(
                agent, question, chat_log_path, config, plugins, logger=logger
            )
            has_issue, reasons = heuristic_critic_has_issue(answer)
            results.append(
                {
                    "question_id": question["question_id"],
                    "attempt": attempt,
                    "answer": answer,
                    "execution_error": answer.startswith(
                        f"Error processing question {qid}:"
                    ),
                    "agent_response": answer,
                    "heuristic_issue": has_issue,
                    "heuristic_reasons": reasons,
                    "heuristic_success": not has_issue,
                    "token_usage": usage,
                    "tool_calls": calls,
                    "tool_output_dir": str(tool_output_dir.resolve()),
                }
            )
            _flush_results(path, results)
        finally:
            clear_tool_outputs(tool_output_dir)

    _flush_results(path, results)
    return {
        "attempt": attempt,
        "results": results,
        "failed_ids": [
            str(item["question_id"]) for item in results if item["heuristic_issue"]
        ],
        "token_usage": {
            key: sum(item["token_usage"][key] for item in results)
            for key in TOKEN_FIELDS
        },
        "plugin_token_usage": dict.fromkeys(TOKEN_FIELDS, 0),
    }


async def _feedback(plugins: list[AgentPlugin], history: dict, chat_log: Path) -> None:
    failed = [item for item in history["results"] if item["heuristic_issue"]]
    history["plugin_token_usage"] = dict.fromkeys(TOKEN_FIELDS, 0)
    for plugin in plugins:
        usage = await plugin.after_attempt(history["attempt"], failed, chat_log)
        if set(usage) != set(TOKEN_FIELDS) or any(
            type(value) is not int or value < 0 for value in usage.values()
        ):
            raise ValueError(f"Invalid token usage from plugin {plugin.name}: {usage}")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError(
                f"Inconsistent token usage from plugin {plugin.name}: {usage}"
            )
        for key in TOKEN_FIELDS:
            history["plugin_token_usage"][key] += usage[key]


def _save_summary(
    output_dir: Path,
    config: ExperimentConfig,
    dataset: Dataset,
    questions: list[dict],
    history: list[dict],
) -> None:
    statistics = compute_statistics(
        aggregate_results([item["results"] for item in history]), history
    )
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        trace_dir / "summary.json",
        {
            "dataset": dataset.name,
            "plugins": [item.name for item in config.plugins],
            "total_questions": len(questions),
            "max_attempts": config.max_attempts,
            "attempts_executed": len(history),
            "statistics": statistics,
            "attempt_summaries": [
                {
                    "attempt": item["attempt"],
                    "questions_evaluated": len(item["results"]),
                    "failed_count": len(item["failed_ids"]),
                    "failed_ids": item["failed_ids"],
                }
                for item in history
            ],
        },
    )
    final = {}
    for execution in history:
        for record in execution["results"]:
            qid = str(record["question_id"])
            if qid not in final:
                final[qid] = {
                    "question_id": qid,
                    "cumulative_tool_calls": [],
                    "output_roots": [],
                    "token_usage": dict.fromkeys(TOKEN_FIELDS, 0),
                }
            question = final[qid]
            question.update(
                answer=record["answer"],
                execution_error=record["execution_error"],
                tool_calls=record["tool_calls"],
            )
            question["cumulative_tool_calls"].extend(record["tool_calls"])
            if record["tool_output_dir"] not in question["output_roots"]:
                question["output_roots"].append(record["tool_output_dir"])
            for key in TOKEN_FIELDS:
                question["token_usage"][key] += record["token_usage"][key]
    _flush_results(output_dir / "results.jsonl", [final[qid] for qid in sorted(final)])
    costs = statistics["token_summary"]
    _write_json(
        output_dir / "experiment_summary.json",
        {
            "dataset": dataset.name,
            "plugins": [item.name for item in config.plugins],
            "total_questions": len(questions),
            "evaluated_questions": len(final),
            "execution_errors": sum(item["execution_error"] for item in final.values()),
            "token_usage": {
                "actor": costs["question_answering_tokens"],
                "plugins": costs["plugin_generation_tokens"],
                "total": costs["grand_total"],
            },
        },
    )


async def run_experiment(
    config: ExperimentConfig,
    dataset: Dataset,
    run_dir: Path,
    *,
    resume_from_dir: Path | None = None,
    continue_from_dir: Path | None = None,
    resume_through_attempt: int = 1,
) -> Path:
    """Store one execution in result/ and its evaluation beside it."""
    questions = dataset.load_questions(config.sample)
    snapshot = {
        **dataset.protocol(),
        "question_ids": [str(item["question_id"]) for item in questions],
    }
    result_dir = run_dir / "result"
    for source in (resume_from_dir, continue_from_dir):
        if source is not None:
            previous = json.loads(
                (source / "result" / "dataset_snapshot.json").read_text(
                    encoding="utf-8"
                )
            )
            if previous != snapshot:
                raise ValueError(
                    "Resume requires the same dataset version and question selection"
                )
    if continue_from_dir is not None:
        if run_dir.resolve() != continue_from_dir.resolve():
            raise ValueError(
                "An in-place continuation must use the selected run directory"
            )
        previous_config = ExperimentConfig.from_yaml(
            result_dir / "experiment_config.yaml"
        )
        if previous_config != config:
            raise ValueError(
                "Continuing in place requires the original run configuration"
            )
        metadata = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        result_dir.mkdir()
        (run_dir / "evaluation").mkdir()
        config.to_yaml(result_dir / "experiment_config.yaml")
        _write_json(result_dir / "dataset_snapshot.json", snapshot)
        metadata = {
            "dataset": dataset.name,
            "model": config.model_name,
            "plugins": [item.name for item in config.plugins],
        }
        if resume_from_dir is not None:
            metadata.update(
                resume_from=str(resume_from_dir),
                resume_through_attempt=resume_through_attempt,
            )
    metadata["status"] = "running"
    _write_json(result_dir / "run.json", metadata)
    try:
        await _execute_experiment(
            config,
            dataset,
            questions,
            result_dir,
            resume_from_dir=resume_from_dir / "result" if resume_from_dir else None,
            continue_from_dir=result_dir if continue_from_dir else None,
            resume_through_attempt=resume_through_attempt,
        )
    except BaseException:
        metadata["status"] = "failed"
        _write_json(result_dir / "run.json", metadata)
        raise
    metadata["status"] = "completed"
    _write_json(result_dir / "run.json", metadata)
    return run_dir


async def _execute_experiment(
    config: ExperimentConfig,
    dataset: Dataset,
    questions: list[dict],
    result_dir: Path,
    *,
    resume_from_dir: Path | None,
    continue_from_dir: Path | None,
    resume_through_attempt: int,
) -> None:
    history = []
    start_attempt = 1
    completed = []
    if resume_from_dir is not None:
        if not 1 <= resume_through_attempt < config.max_attempts:
            raise ValueError("resume-through-attempt must precede max_attempts")
        history, _, _ = load_attempts_through(resume_from_dir, resume_through_attempt)
        seed_resume_files(resume_from_dir, result_dir, resume_through_attempt)
        start_attempt = resume_through_attempt + 1
    if continue_from_dir is not None:
        attempt_path = result_dir / "traces" / "attempt_1.jsonl"
        if any(
            int(path.stem.removeprefix("attempt_")) > 1
            for path in (result_dir / "traces").glob("attempt_*.jsonl")
        ):
            raise ValueError(
                "--continue-attempt is for partial first attempts; use --resume-from for later rounds"
            )
        if attempt_path.exists():
            completed, errors = load_completed_results(attempt_path)
            missing = {str(item["question_id"]) for item in questions} - {
                str(item["question_id"]) for item in completed
            }
            cleanup_chat_after_continue(result_dir / "run.chat", missing | set(errors))
            for log in result_dir.glob("*_langchain.log"):
                cleanup_log_after_continue(log, missing | set(errors))
    plugins = create_plugins(config.plugins)
    clear_tool_outputs(result_dir / "out")
    llm, servers = load_config(config, result_dir, dataset)
    client = None
    logger = create_experiment_logger(
        result_dir, config.model_name, autoplanning=config.autoplanning
    )
    chat_log = result_dir / "run.chat"
    print(
        f"Dataset: {dataset.name} | Model: {config.model_name} | Plugins: {config.method_label} | Output: {result_dir}"
    )
    try:
        for plugin in plugins:
            await plugin.setup(PluginContext(config, result_dir, llm))
        agent, client = await create_agent(llm, servers)
        if history and history[-1]["failed_ids"]:
            await _feedback(plugins, history[-1], chat_log)
        for attempt in range(start_attempt, config.max_attempts + 1):
            if history and not history[-1]["failed_ids"]:
                break
            selected = (
                questions
                if attempt == 1
                else [
                    item
                    for item in questions
                    if str(item["question_id"]) in set(history[-1]["failed_ids"])
                ]
            )
            result = await run_single_attempt(
                agent,
                selected,
                attempt,
                chat_log,
                config,
                plugins,
                result_dir,
                logger,
                completed,
            )
            completed = []
            history.append(result)
            _save_summary(result_dir, config, dataset, questions, history)
            if result["failed_ids"] and attempt < config.max_attempts:
                await _feedback(plugins, result, chat_log)
                _save_summary(result_dir, config, dataset, questions, history)
        _save_summary(result_dir, config, dataset, questions, history)
    finally:
        try:
            if client is not None and hasattr(client, "close"):
                result = client.close()
                if inspect.isawaitable(result):
                    await result
        finally:
            try:
                clear_tool_outputs(result_dir / "out")
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)

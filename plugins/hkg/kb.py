"""Knowledge-base parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path


def parse_summary_to_subtasks(summary_text: str, domain: str) -> list:
    """Parse summary text into subtasks for retrieval (mirrors scripting utility)."""
    if not isinstance(summary_text, str) or not summary_text.strip():
        return []

    split_pattern = r"(?<=\n)\s*(?:\*|-|\+|\*\*[^\n]+\*\*)\s*"
    raw_items = re.split(split_pattern, summary_text.strip())
    raw_items = [item.strip() for item in raw_items if item.strip()]

    if raw_items and not re.match(r"^\*\*[^\n]+\*\*", raw_items[0]):
        first_item = raw_items[0]
        title_match = re.search(r"\*\*(.+?)\*\*:", first_item)
        if title_match:
            title_end = title_match.end()
            first_subtask = first_item[:title_end] + first_item[title_end:].split("\n")[0].strip()
            raw_items[0] = first_subtask
            remaining = first_item[title_end:].split("\n")[1:]
            if remaining:
                raw_items = [first_subtask] + [r.strip() for r in remaining if r.strip()]

    subtasks = []
    for item in raw_items:
        if not item:
            continue

        title_match = re.search(r"\*\*([^*]+?)\*\*", item)
        sub_title = title_match.group(1).strip() if title_match else "General Analysis"

        content = re.sub(r"\*\*[^*]+?\*\*", "", item).strip()
        content = re.sub(r"^[:\-*+]\s*", "", content).strip()
        content = re.sub(r"\s+", " ", content)

        if not content:
            content = item

        subtasks.append(
            {
                "domain": domain,
                "sub_title": sub_title,
                "content": content,
                "full_text": f"[{domain}] {sub_title}: {content}",
            }
        )

    return subtasks


def load_knowledge_base(file_path: str) -> dict:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not all(isinstance(value, list) for value in data.values()):
        raise ValueError("HKG knowledge must be a non-empty mapping of domain names to task lists")
    return data

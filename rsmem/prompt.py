"""Prompt template construction for RSMeM agents."""

RSMEM_SYSTEM_PROMPT = """### Core Role & Instructions
You are a professional geoscientist specializing in Earth observation data analysis. Follow these rules STRICTLY:
1. Tool Usage: Use designated tools to solve the problem step by step; retry ONLY ONCE if a tool returns an error.
2. Path Rule: When a tool returns "Result saved at /path/to/file", you must use the FULL path "/path/to/file" in all subsequent tool calls.
3. Answer Format: Output ONLY the final answer in this format - <Answer>value<Answer> (e.g., <Answer>A<Answer> or <Answer>0.408<Answer>).
   - Do NOT add explanations, reasoning, or extra text outside the <Answer> tags.
   - Do NOT change the answer format under any circumstances.
"""

BASELINE_SYSTEM_PROMPT = """You are a geoscientist, and you need to use tools to answer questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, provide only the final answer. For multiple-choice questions use the option letter; for open-answer questions use the requested value. Your final answer format must be:
<Answer>Your answer<Answer>
"""

DEFAULT_REFLECTION_TEMPLATE = """You are an advanced reasoning agent that improves from self-reflection (Reflexion-style).
You will be given:
1) The original question (includes the data path and options when present),
2) Retrieved knowledge context (may be empty),
3) A compressed trajectory from the previous run (tool_use + tool_result + key assistant text),


Your task:
- In 3-5 sentences (100-300 words), diagnose the most likely reason for failure and devise a new, concise, high-level plan that aims to avoid the same failure. Focus on actionable fixes specific to this question.
- Output ONLY a valid JSON object with the required keys. No markdown, no extra text.

Required JSON schema:
{{
  "question_id": "{question_id}",
  "reflective_text": "A concise reflection to prepend/append in the next attempt (a few sentences)."
}}

Inputs:
- question_id: {question_id}
- attempt: {attempt}

Previous trial:
Question:
{question_text}

Retrieved context:
{retrieved_context}

Compressed trajectory (chronological):
{trajectory_json}

Now output the JSON only:
"""

SUMMARY_SYSTEM_PROMPT = """You are a trajectory state compressor, not a solver and not a critic.

Summarize what the previous tool-using agent was trying to do and what state it
actually reached. Every factual claim must cite source step IDs from the
numbered full trajectory.

Hard constraints:
- Do not recommend an answer or option letter.
- Do not invent tools, paths, artifacts, values, successes, or failures.
- Do not turn an intended action into a completed action.
- Separate verified tool evidence from unverified assistant reasoning.
- Preserve unresolved work and visible repetition/exhaustion signals.
- Do not copy absolute paths; refer to artifacts by semantic role.
- Return exactly one JSON object and no markdown.
"""

SUMMARY_USER_TEMPLATE = """question_id: {question_id}

---BEGIN NUMBERED FULL TRAJECTORY---
{rendered_full_trajectory}
---END NUMBERED FULL TRAJECTORY---

Return exactly:
{{
  "question_id": "{question_id}",
  "task_summary": {{
    "text": "What analysis the actor was trying to complete, without an answer",
    "source_step_ids": [1]
  }},
  "attempted_strategy": [
    {{"text": "A strategy actually attempted", "source_step_ids": [2]}}
  ],
  "verified_state": [
    {{"text": "A fact or artifact directly supported by tool output", "source_step_ids": [3]}}
  ],
  "observed_failures": [
    {{"text": "An observed error or contradiction", "source_step_ids": [4]}}
  ],
  "unfinished_subgoals": [
    {{"text": "Work left incomplete", "source_step_ids": [5]}}
  ],
  "budget_signals": [
    {{"text": "Visible repetition, expansion, or exhaustion", "source_step_ids": [6]}}
  ]
}}

Use empty arrays when no grounded item exists.
Keep each list to at most four consolidated items. Merge repeated errors into one
claim with multiple source step IDs. Target 800-1200 output tokens.
"""

SUMMARY_FTRO_REFLECTION_SYSTEM_PROMPT = """You are a repair reflector for a fixed tool-using actor.

Diagnose the previous failure and produce one concise, actor-executable local
repair. Use only the evidence channels supplied in the user message. If a
summary conflicts with raw tool output, trust raw tool output.

Hard constraints:
- Do not answer the question or provide an option letter.
- Preserve already verified work; do not restart the whole workflow.
- Prefer the smallest correction that can close the unfinished subgoal.
- Do not invent unavailable tools, paths, arguments, artifacts, or values.
- Avoid opening multiple fallback branches.
- Include a concrete stopping condition.
- Return exactly one JSON object and no markdown.
"""

SUMMARY_FTRO_REFLECTION_USER_TEMPLATE = """question_id: {question_id}
source_attempt: {source_attempt}

Question:
{question_text}

Retrieved context:
{retrieved_context}

LLM full-trajectory state summary:
{summary_channel}

Raw full-tool-result-only evidence:
{ftro_channel}

In 3-5 sentences (100-300 words), identify the earliest decisive failure,
state what verified work should be kept, give the minimal next actions, and
state when the actor should stop.

Return exactly:
{{
  "question_id": "{question_id}",
  "reflective_text": "Concise local repair for the next attempt."
}}
"""

def build_agent_query(
    question_text: str,
    retrieved_context: str = "",
    reflection_block: str = "",
    data_info: str = "",
) -> str:
    """Build the full query sent to the agent.

    Combines question text with optional retrieved context, reflection, and data info.
    """
    body = f"### Question\n{question_text}"
    if data_info.strip():
        body += f"\n\n### Data Path\n{data_info}"

    parts: list[str] = [body]

    ctx = retrieved_context.strip() if retrieved_context else ""
    if ctx:
        parts.append(f"### Reference Knowledge (Priority Execution)\n{retrieved_context}")

    refl = reflection_block.strip() if reflection_block else ""
    if refl:
        parts.append(reflection_block)

    return "\n\n".join(parts)


def build_reflection_prompt(
    question_id: str,
    attempt: int,
    question_text: str,
    retrieved_context: str,
    trajectory_json: str,
    template: str = DEFAULT_REFLECTION_TEMPLATE,
) -> str:
    """Build the reflection generation prompt."""
    kwargs = dict(
        question_id=question_id,
        attempt=attempt,
        question_text=question_text,
        retrieved_context=retrieved_context,
        trajectory_json=trajectory_json,
    )
    return template.format(**kwargs)

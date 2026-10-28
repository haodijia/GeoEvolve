"""Hierarchical Knowledge Grounding as an optional prompt component."""

from dataclasses import dataclass

from plugins.base import AgentPlugin, PluginContext, PromptContribution
from plugins.hkg.kb import load_knowledge_base
from plugins.hkg.retrieval import (
    EmbeddingModel,
    precompute_kb_embeddings,
    hybrid_retrieval,
)


@dataclass
class HKGConfig:
    knowledge: str = "plugins/hkg/knowledge.json"
    model: str = "plugins/hkg/models/all-mpnet-base-v2"


class HKGPlugin(AgentPlugin):
    def __init__(self, options: dict) -> None:
        self.config = HKGConfig(**options)

    async def setup(self, context: PluginContext) -> None:
        await super().setup(context)
        self.model = EmbeddingModel(self.config.model)
        self.embeddings = precompute_kb_embeddings(
            load_knowledge_base(self.config.knowledge), self.model
        )
        if not self.embeddings:
            raise ValueError("HKG knowledge base contains no retrievable subtasks")

    async def before_question(self, question: dict, query: str) -> PromptContribution:
        results = hybrid_retrieval(query, self.embeddings, self.model)
        if not results:
            return PromptContribution()
        result = results[0]
        context = (
            f"**Domain**: {result['domain']}\n"
            f"**Sub-task**: {result['sub_title']}\n"
            f"**Core Methodology**: {result['summary'][:1000].strip()}\n"
            "**Execution Rule**: Prioritize the above logic while dynamically adjusting "
            "file paths and parameters according to the current environment."
        )
        return PromptContribution(
            f"### Reference Knowledge (Priority Execution)\n{context}",
            {"retrieved_context": context},
        )

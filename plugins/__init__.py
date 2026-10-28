"""Load built-in or explicitly named local components without importing their dependencies."""

from importlib import import_module

from agent.config import PluginConfig
from plugins.base import AgentPlugin

BUILTINS = {
    "hkg": "plugins.hkg.plugin:HKGPlugin",
    "far": "plugins.far.plugin:FARPlugin",
}


def create_plugins(specs: list[PluginConfig]) -> list[AgentPlugin]:
    instances = []
    for spec in specs:
        target = BUILTINS.get(spec.name, spec.name)
        if ":" not in target:
            raise ValueError(
                f"Unknown plugin: {spec.name}; use hkg, far or module:Class"
            )
        module, name = target.split(":", 1)
        factory = getattr(import_module(module), name)
        if not isinstance(factory, type) or not issubclass(factory, AgentPlugin):
            raise TypeError(f"{target} must be an AgentPlugin subclass")
        plugin = factory(spec.options)
        plugin.name = spec.name
        instances.append(plugin)
    return instances

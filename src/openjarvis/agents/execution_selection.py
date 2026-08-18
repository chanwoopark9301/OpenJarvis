"""Select the execution agent class for a resolved toolkit."""


def select_execution_agent_class(configured_cls: type, *, has_tools: bool) -> type:
    if not has_tools or getattr(configured_cls, "accepts_tools", False):
        return configured_cls
    if not getattr(configured_cls, "supports_managed_tool_fallback", False):
        return configured_cls
    from openjarvis.agents.orchestrator import OrchestratorAgent

    return OrchestratorAgent

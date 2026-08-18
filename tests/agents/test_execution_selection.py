from openjarvis.agents.execution_selection import select_execution_agent_class
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.agents.simple import SimpleAgent


def test_simple_agent_uses_native_tool_loop_when_tools_exist():
    assert select_execution_agent_class(SimpleAgent, has_tools=True) is OrchestratorAgent


def test_simple_agent_stays_simple_without_tools():
    assert select_execution_agent_class(SimpleAgent, has_tools=False) is SimpleAgent


def test_non_opted_in_agent_is_never_replaced():
    class Specialized:
        accepts_tools = False
        supports_managed_tool_fallback = False

    assert select_execution_agent_class(Specialized, has_tools=True) is Specialized

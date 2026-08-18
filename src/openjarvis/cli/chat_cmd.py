"""``jarvis chat`` — interactive multi-turn chat REPL."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.markdown import Markdown

from openjarvis.cli._tool_names import resolve_tool_names
from openjarvis.core.config import (
    canonical_profile_is_active,
    effective_chat_memory_files,
    load_config,
)
from openjarvis.core.events import EventBus
from openjarvis.core.types import Message, Role
from openjarvis.memory import record_and_publish_completed_exchange

logger = logging.getLogger(__name__)


def _read_input(prompt: str = "You> ") -> Optional[str]:
    """Read user input with graceful EOF handling."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _personal_memory_status(config, personal_mode: str) -> str:
    """Describe whether stored personal memory can influence responses."""
    if not config.agent.context_from_memory:
        return "collecting; response injection disabled"
    if personal_mode == "shadow":
        return "collecting; direct rule injection enabled"
    return "active"


def _build_chat_agent_context(
    history: list[Message], dynamic_messages: list[Message]
):
    """Build an agent context from dynamic memory and prior dialogue."""
    from openjarvis.agents._stubs import AgentContext

    context = AgentContext()
    for message in dynamic_messages:
        context.conversation.add(message)
    for message in history:
        if message.role != Role.SYSTEM:
            context.conversation.add(message)
    return context


def _pending_user_dialogue_messages(
    personal_context: object | None,
    *,
    local_response_engine: bool,
) -> list[Message]:
    """Project local pending user turns as dialogue, never instructions."""
    if not local_response_engine or personal_context is None:
        return []
    return [
        Message(
            role=Role.USER,
            content=text,
            metadata={"personal_memory_context": "recent_dialogue"},
        )
        for text in getattr(personal_context, "recent_pending_user_messages", ())
        if str(text).strip()
    ]


def _prepend_pending_dialogue(
    messages: list[Message],
    pending: list[Message],
) -> list[Message]:
    """Place restart dialogue after system identity and before live history."""
    if not pending:
        return messages
    system = [message for message in messages if message.role == Role.SYSTEM]
    dialogue = [message for message in messages if message.role != Role.SYSTEM]
    return [*system, *pending, *dialogue]


@click.command()
@click.option("-e", "--engine", "engine_key", default=None, help="Engine backend.")
@click.option("-m", "--model", "model_name", default=None, help="Model to use.")
@click.option("-a", "--agent", "agent_name", default=None, help="Agent type.")
@click.option("--tools", default=None, help="Comma-separated tool names.")
@click.option("--system", "system_prompt", default=None, help="Custom system prompt.")
@click.option(
    "--persona",
    "persona_name",
    default=None,
    help=(
        "Named persona dir under ~/.openjarvis/personas/<name>/ "
        "(overrides config). Pass 'none' to disable all persona files."
    ),
)
def chat(
    engine_key: str | None,
    model_name: str | None,
    agent_name: str | None,
    tools: str | None,
    system_prompt: str | None,
    persona_name: str | None,
) -> None:
    """Start an interactive multi-turn chat session.

    Commands during chat:
      /quit, /exit  — end session
      /clear        — clear conversation history
      /model        — show current model
      /help         — show available commands
      /history      — show conversation history
    """
    console = Console(stderr=True)

    config = load_config()
    bus = EventBus(record_history=False)

    import dataclasses as _dc

    requested_mf = (
        _dc.replace(config.memory_files, persona_name=persona_name)
        if persona_name is not None
        else config.memory_files
    )

    # Resolve engine
    from openjarvis.engine import get_engine
    from openjarvis.intelligence import register_builtin_models

    register_builtin_models()

    resolved = get_engine(config, engine_key)
    if resolved is None:
        console.print("[red]No inference engine available.[/red]")
        sys.exit(1)

    engine_name, engine = resolved
    from openjarvis.memory.candidate_extractor import (
        is_local_personal_memory_engine,
    )

    local_response_engine = is_local_personal_memory_engine(config, engine_name)
    canonical_profile_active = canonical_profile_is_active(config)
    effective_mf = effective_chat_memory_files(
        config,
        requested_mf,
        personal_context_allowed=local_response_engine,
    )
    model = model_name or config.intelligence.default_model
    if not model:
        from openjarvis.engine import discover_engines, discover_models

        all_engines = discover_engines(config)
        all_models = discover_models(all_engines)
        engine_models = all_models.get(engine_name, [])
        if engine_models:
            model = engine_models[0]
        else:
            console.print("[red]No model available.[/red]")
            sys.exit(1)

    if (
        config.personal_memory.enabled
        and config.personal_memory.mode != "off"
        and not hasattr(engine, "is_generating")
    ):
        from openjarvis.telemetry.instrumented_engine import InstrumentedEngine

        engine = InstrumentedEngine(engine, bus)

    # Resolve agent (optional)
    agent = None
    agent_key = agent_name or config.agent.default_agent
    if agent_key and agent_key != "none":
        try:
            import openjarvis.agents  # noqa: F401 — trigger registration
            from openjarvis.core.registry import AgentRegistry

            if AgentRegistry.contains(agent_key):
                configured_cls = AgentRegistry.get(agent_key)
                kwargs: dict = {"bus": bus}
                tool_instances = []

                if (
                    getattr(configured_cls, "accepts_tools", False)
                    or getattr(
                        configured_cls, "supports_managed_tool_fallback", False
                    )
                ):
                    tool_names_list = resolve_tool_names(
                        tools,
                        getattr(config.tools, "enabled", None),
                        getattr(config.agent, "tools", None),
                    )
                    if tool_names_list:
                        import openjarvis.tools  # noqa: F401 — trigger registration
                        from openjarvis.core.registry import ToolRegistry
                        from openjarvis.tools._stubs import BaseTool

                        for tname in tool_names_list:
                            if ToolRegistry.contains(tname):
                                tcls = ToolRegistry.get(tname)
                                if isinstance(tcls, type) and issubclass(
                                    tcls, BaseTool
                                ):
                                    tool_instances.append(tcls())
                                elif isinstance(tcls, BaseTool):
                                    tool_instances.append(tcls)
                from openjarvis.agents.execution_selection import (
                    select_execution_agent_class,
                )

                execution_cls = select_execution_agent_class(
                    configured_cls,
                    has_tools=bool(tool_instances),
                )
                logger.debug(
                    "Chat agent %r selected execution class %s",
                    agent_key,
                    execution_cls.__name__,
                )

                if getattr(execution_cls, "accepts_tools", False):
                    kwargs["tools"] = tool_instances
                    kwargs["max_turns"] = config.agent.max_turns

                    def _confirm(prompt: str) -> bool:
                        console.print(
                            f"[yellow]Confirm:[/yellow] {prompt} [y/N] ",
                            end="",
                        )
                        ans = input().strip().lower()
                        return ans in ("y", "yes")

                    kwargs["interactive"] = True
                    kwargs["confirm_callback"] = _confirm

                import inspect as _inspect

                if (
                    "prompt_builder"
                    in _inspect.signature(execution_cls.__init__).parameters
                ):
                    from openjarvis.prompt.builder import SystemPromptBuilder

                    kwargs["prompt_builder"] = SystemPromptBuilder(
                        agent_template=config.agent.default_system_prompt or "",
                        memory_files_config=effective_mf,
                        system_prompt_config=config.system_prompt,
                    )

                agent = execution_cls(engine, model, **kwargs)
        except Exception as exc:
            console.print(f"[yellow]Agent '{agent_key}' failed: {exc}[/yellow]")

    # Print banner
    console.print(
        f"[green bold]OpenJarvis Chat[/green bold]\n"
        f"  Engine: [cyan]{engine_name}[/cyan]  Model: [cyan]{model}[/cyan]"
        f"  Agent: [cyan]{agent_key or 'direct'}[/cyan]\n"
        f"  Type /help for commands, /quit to exit.\n"
    )

    # Background-work status banner (disappears after first user message)
    from openjarvis.cli._bg_state import get_status
    from openjarvis.cli._chat_banner import render_startup_banner

    _banner = render_startup_banner(get_status())
    if _banner:
        console.print(f"[dim cyan]{_banner}[/dim cyan]")

    # Completion-notification dispatcher (fires once per task per session)
    from openjarvis.cli._chat_notifications import NotificationDispatcher

    _notifications = NotificationDispatcher(get_status())

    # Automatic long-term memory — extracts durable facts in the background.
    memory_service = None
    try:
        from openjarvis.memory import build_memory_service

        memory_service = build_memory_service(config, engine, model, event_bus=bus)
        if memory_service is not None:
            memory_service.start()
            console.print("[dim]  Memory: active[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Memory service unavailable: {exc}[/yellow]")
        memory_service = None

    personal_memory_service = None
    try:
        from openjarvis.memory import build_personal_memory_service

        personal_memory_service = build_personal_memory_service(
            config,
            engine,
            engine_key=engine_name,
            default_model=model,
            event_bus=bus,
        )
        if personal_memory_service is not None:
            personal_memory_service.start()
            personal_mode = getattr(config.personal_memory, "mode", "active")
            personal_status = _personal_memory_status(config, personal_mode)
            console.print(f"[dim]  Personal memory: {personal_status}[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Personal memory unavailable: {exc}[/yellow]")
        personal_memory_service = None

    # The document backend and automatic fact store are separate persistence
    # mechanisms. Context injection combines both at read time so facts from
    # previous sessions are immediately available without a manual index step.
    memory_backend = None
    if config.agent.context_from_memory:
        from openjarvis.cli.ask import _get_memory_backend

        memory_backend = _get_memory_backend(config)

    # Conversation state
    if not system_prompt:
        from openjarvis.prompt.builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            agent_template=config.agent.default_system_prompt or "",
            memory_files_config=effective_mf,
            system_prompt_config=config.system_prompt,
        )
        system_prompt = builder.build()

    history: List[Message] = []
    if system_prompt:
        history.append(Message(role=Role.SYSTEM, content=system_prompt))
    # REPL loop
    while True:
        for note in _notifications.diff(get_status()):
            console.print(f"[dim cyan]{note}[/dim cyan]")

        user_input = _read_input()
        if user_input is None:
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif cmd == "/clear":
            history = []
            if system_prompt:
                history.append(Message(role=Role.SYSTEM, content=system_prompt))
            console.print("[dim]History cleared.[/dim]")
            continue
        elif cmd == "/model":
            console.print(
                f"Model: [cyan]{model}[/cyan]  Engine: [cyan]{engine_name}[/cyan]"
            )
            continue
        elif cmd == "/help":
            console.print(
                "[bold]Commands:[/bold]\n"
                "  /quit, /exit  — end session\n"
                "  /clear        — clear conversation\n"
                "  /model        — show model info\n"
                "  /history      — show conversation\n"
                "  /help         — this message"
            )
            continue
        elif cmd == "/history":
            if not history:
                console.print("[dim]No history yet.[/dim]")
            else:
                for msg in history:
                    role_str = msg.role if isinstance(msg.role, str) else msg.role.value
                    role = role_str.upper()
                    console.print(f"[bold]{role}:[/bold] {msg.content[:200]}")
            continue

        # Add user message
        history.append(Message(role=Role.USER, content=user_input))

        agent_context_messages: list[Message] = []
        generation_history = history
        if config.agent.context_from_memory:
            try:
                from openjarvis.memory import load_configured_facts
                from openjarvis.memory.context_composer import (
                    compose_configured_personal_context,
                )
                from openjarvis.tools.storage.context import (
                    ContextConfig,
                    inject_context,
                )

                if memory_service is not None and hasattr(memory_service, "list_facts"):
                    facts = memory_service.list_facts()
                else:
                    facts = load_configured_facts(config)
                from openjarvis.memory.store import legacy_fact_context_enabled

                if (
                    not local_response_engine
                    or canonical_profile_active
                    or not legacy_fact_context_enabled(config)
                ):
                    facts = []
                ctx_cfg = ContextConfig(
                    top_k=config.memory.context_top_k,
                    min_score=config.memory.context_min_score,
                    max_context_tokens=config.memory.context_max_tokens,
                )
                personal_context = None
                if local_response_engine:
                    personal_context = compose_configured_personal_context(
                        config,
                        user_input,
                        engine_key=engine_name,
                        archive=(
                            personal_memory_service.archive
                            if personal_memory_service is not None
                            else None
                        ),
                    )
                pending_dialogue = _pending_user_dialogue_messages(
                    personal_context,
                    local_response_engine=local_response_engine,
                )
                base_messages = [] if agent is not None else history
                if agent is None:
                    base_messages = _prepend_pending_dialogue(
                        base_messages,
                        pending_dialogue,
                    )
                context_messages = inject_context(
                    user_input,
                    base_messages,
                    memory_backend,
                    config=ctx_cfg,
                    facts=facts,
                    personal_context=personal_context,
                )
                if agent is not None:
                    agent_context_messages.extend(context_messages)
                    agent_context_messages.extend(pending_dialogue)
                else:
                    generation_history = context_messages
            except Exception:
                logger.debug("Failed to inject memory context", exc_info=True)

        # Generate response even when optional memory context is unavailable.
        try:
            if agent is not None:
                agent_context = _build_chat_agent_context(
                    history[:-1], agent_context_messages
                )
                response = agent.run(user_input, context=agent_context)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )
            else:
                result = engine.generate(generation_history, model=model)
                content = (
                    result.get("content", "")
                    if isinstance(result, dict)
                    else str(result)
                )

            history.append(Message(role=Role.ASSISTANT, content=content))
            console.print()
            console.print(Markdown(content))
            console.print()

            record_and_publish_completed_exchange(
                bus,
                personal_memory_service,
                user_input,
                content,
                source="cli.chat",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Generation interrupted.[/dim]")
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]\n")

    if memory_service is not None:
        memory_service.stop()
    if personal_memory_service is not None:
        personal_memory_service.stop()


__all__ = ["chat"]

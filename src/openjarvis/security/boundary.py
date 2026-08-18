"""BoundaryGuard — scans content at device exit points.

Wraps SecretScanner and PIIScanner to redact, warn, or block
secrets and PII before data leaves the device via cloud engines
or external tool calls.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

from openjarvis.core.types import ToolCall

if TYPE_CHECKING:
    from openjarvis.core.events import EventBus
    from openjarvis.security._stubs import BaseScanner

logger = logging.getLogger(__name__)


class SecurityBlockError(Exception):
    """Raised when mode='block' and secrets/PII are detected."""


class BoundaryGuard:
    """Scans outbound content for secrets and PII at device boundaries.

    Parameters
    ----------
    mode:
        Action on findings: ``"redact"`` replaces matches,
        ``"warn"`` logs but passes through, ``"block"`` raises.
    enabled:
        Master switch. When ``False``, all content passes through.
    bus:
        Optional event bus for publishing SECURITY_ALERT events.
    scanners:
        Custom scanners. Defaults to SecretScanner + PIIScanner.
    """

    def __init__(
        self,
        mode: str = "redact",
        *,
        enabled: bool = True,
        bus: Optional["EventBus"] = None,
        scanners: Optional[List["BaseScanner"]] = None,
    ) -> None:
        self._mode = mode
        self._enabled = enabled
        self._bus = bus
        if scanners is not None:
            self._scanners = scanners
        else:
            self._scanners = self._default_scanners()

    @staticmethod
    def _default_scanners() -> List["BaseScanner"]:
        try:
            from openjarvis.security.scanner import PIIScanner, SecretScanner

            return [SecretScanner(), PIIScanner()]
        except (ImportError, Exception) as exc:
            logger.warning(
                "Rust-backed scanners unavailable (%s); "
                "BoundaryGuard running without scanners. "
                "Build the Rust extension: uv run maturin develop",
                exc,
            )
            return []

    def scan_outbound(self, content: str, destination: str) -> str:
        """Scan text before it leaves the device.

        Returns redacted text in ``"redact"`` mode, original text in
        ``"warn"`` mode, or raises ``SecurityBlockError`` in ``"block"``
        mode when findings are detected.
        """
        if not self._enabled or not content:
            return content

        has_findings = False
        redacted = content
        for scanner in self._scanners:
            result = scanner.scan(content)
            if result.findings:
                has_findings = True
                if self._mode == "redact":
                    redacted = scanner.redact(redacted)

        if has_findings:
            self._emit_alert(destination, content)
            if self._mode == "block":
                raise SecurityBlockError(
                    f"Secrets/PII detected in outbound content to {destination}"
                )
            if self._mode == "warn":
                logger.warning(
                    "Secrets/PII detected in outbound content to %s", destination
                )
                return content
            return redacted

        return content

    def check_outbound(
        self,
        tool_call: ToolCall,
        *,
        private_context: Sequence[str] = (),
        public_context: str = "",
    ) -> ToolCall:
        """Scan tool call arguments before execution.

        Returns a new ToolCall with redacted arguments if needed.
        """
        if not self._enabled or not tool_call.arguments:
            return tool_call

        destination = f"tool:{tool_call.name}"
        if self._contains_private_context(
            tool_call.arguments,
            private_context,
            public_context,
        ):
            self._emit_alert(destination, tool_call.arguments)
            raise SecurityBlockError(
                f"Private context detected in outbound content to {destination}"
            )

        redacted_args = self.scan_outbound(
            tool_call.arguments,
            destination=destination,
        )
        if redacted_args != tool_call.arguments:
            return replace(tool_call, arguments=redacted_args)
        return tool_call

    @classmethod
    def _contains_private_context(
        cls,
        arguments: str,
        private_context: Sequence[str],
        public_context: str = "",
    ) -> bool:
        """Detect copied private prompt data without inferring user intent.

        Only turn-local strings that were actually supplied as private prompt
        context are considered. Matching is Unicode-normalized and requires a
        substantive contiguous fragment, so an independent current-user query
        such as a public location/weather lookup remains eligible for tools.
        """
        if not private_context:
            return False
        outbound_tokens = cls._normalized_tokens(cls._argument_text(arguments))
        if not outbound_tokens:
            return False
        outbound = f" {' '.join(outbound_tokens)} "
        public_tokens = cls._normalized_tokens(public_context)
        public = f" {' '.join(public_tokens)} "
        for source in private_context:
            for fragment in cls._private_fragments(str(source)):
                normalized = f" {' '.join(fragment)} "
                if normalized in outbound and normalized not in public:
                    return True
        return False

    @staticmethod
    def _argument_text(arguments: str) -> str:
        """Return decoded string values from JSON arguments for comparison."""
        try:
            decoded = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return arguments

        def strings(value: object) -> Iterable[str]:
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        return "\n".join(strings(decoded))

    @staticmethod
    def _normalized_tokens(text: str) -> tuple[str, ...]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return tuple(re.findall(r"[\w가-힣]+", normalized, flags=re.UNICODE))

    @classmethod
    def _private_fragments(cls, source: str) -> Iterable[tuple[str, ...]]:
        """Yield deterministic, substantive token windows from private text."""
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stripped = re.sub(r"^\s*(?:[-*+] |\d+[.)]\s+)", "", stripped)
            tokens = cls._normalized_tokens(stripped)
            if len(tokens) > 3:
                for index in range(len(tokens) - 2):
                    yield tokens[index : index + 3]
            elif tokens:
                yield tokens

    def _emit_alert(self, destination: str, content: str) -> None:
        if self._bus is None:
            return
        try:
            from openjarvis.core.events import EventType

            self._bus.publish(
                EventType.SECURITY_ALERT,
                {
                    "source": "boundary_guard",
                    "destination": destination,
                    "mode": self._mode,
                    "content_preview": content[:80],
                },
            )
        except Exception:
            logger.debug("Failed to emit security alert event", exc_info=True)

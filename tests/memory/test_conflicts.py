"""Tests for bounded disequilibrium and reflection triggers."""

from __future__ import annotations

from openjarvis.memory.conflicts import ConflictMonitor, ConflictObservation


def _conflict(session_id: str, schema_id: str = "running"):
    return ConflictObservation(
        schema_id=schema_id,
        session_id=session_id,
        prediction_error=0.8,
        created_at=1.0,
    )


def test_same_schema_conflict_across_two_sessions_triggers_reflection():
    """Ignoring repeated cross-session conflict would freeze a wrong schema."""
    result = ConflictMonitor().should_reflect(
        [_conflict("monday"), _conflict("friday")]
    )

    assert result.triggered is True
    assert "repeated_cross_session_conflict" in result.reason_codes


def test_one_transient_conflict_stays_unresolved_without_reflection():
    """One unusual day must not launch an expensive personality rewrite."""
    result = ConflictMonitor().should_reflect([_conflict("monday")])

    assert result.triggered is False
    assert result.reason_codes == ()


def test_support_counter_balance_triggers_review_without_a_sum():
    """Separate opposing evidence masses must trigger review near equilibrium."""
    result = ConflictMonitor().should_reflect(
        [],
        support_mass=0.55,
        counter_mass=0.48,
    )

    assert result.triggered is True
    assert result.reason_codes == ("support_counter_balance",)


def test_direct_feedback_and_explicit_analysis_are_immediate_triggers():
    """User rejection or a direct why-question must not wait for recurrence."""
    monitor = ConflictMonitor()

    rejected = monitor.should_reflect([], user_rejected=True)
    correction = monitor.should_reflect([], direct_correction=True)
    requested = monitor.should_reflect([], explicit_analysis_request=True)

    assert rejected.reason_codes == ("user_rejected_used_schema",)
    assert correction.reason_codes == ("direct_correction_needs_neighbor_review",)
    assert requested.reason_codes == ("user_requested_analysis",)

"""Tests for loopback-only personal-memory management routes."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind
from openjarvis.memory.social_reflection import ExternalAction
from openjarvis.memory.understanding_gap import GapRoute
from openjarvis.server.personal_memory_routes import router


def _app_with_claim(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    text = "Do not proactively mention timers."
    archive.record_exchange(
        exchange_id="e1",
        user_text=text,
        assistant_text="Acknowledged.",
        source="test",
    )
    assert archive.claim_candidate_job("e1") is not None
    candidate = archive.complete_candidate_job(
        "e1",
        [
            CandidateDraft(
                CandidateKind.CONSTRAINT,
                text,
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="topic:timer",
            )
        ],
        engine_id="ollama",
        extractor_version="test",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied
    app = FastAPI()
    app.state.personal_memory_service = SimpleNamespace(
        archive=archive,
        route_understanding_gap=lambda kind, **kwargs: GapRoute(
            "ask_user" if kind == "intent_ambiguity" else "hold_local",
            "user_is_authority_on_intent",
        ),
        prepare_external_reflection=lambda query, **kwargs: ExternalAction(
            "hold_local",
            "analysis_not_requested",
        ),
    )
    app.include_router(router)
    return app, archive.get_active_claims()[0]


def test_list_explain_and_suppress_from_loopback(tmp_path):
    app, claim = _app_with_claim(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 50000))

    listed = client.get("/v1/personal-memory")
    explained = client.get(f"/v1/personal-memory/{claim.id}")
    suppressed = client.post(
        f"/v1/personal-memory/{claim.id}/suppress",
        json={"reason": "user requested"},
    )

    assert listed.status_code == 200
    assert listed.json()[0]["content"] == claim.content
    assert explained.json()["supporting_evidence"] == [claim.content]
    assert suppressed.status_code == 200


def test_every_route_rejects_non_loopback_client(tmp_path):
    app, claim = _app_with_claim(tmp_path)
    client = TestClient(app, client=("203.0.113.9", 50000))

    assert client.get("/v1/personal-memory").status_code == 403
    assert client.get(f"/v1/personal-memory/{claim.id}").status_code == 403
    assert (
        client.post(
            f"/v1/personal-memory/{claim.id}/suppress",
            json={"reason": "remote request"},
        ).status_code
        == 403
    )


def test_bulk_delete_requires_exact_token(tmp_path):
    app, _claim = _app_with_claim(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 50000))

    rejected = client.request(
        "DELETE",
        "/v1/personal-memory",
        json={"confirm_token": "wrong"},
    )
    accepted = client.request(
        "DELETE",
        "/v1/personal-memory",
        json={"confirm_token": "DELETE ALL PERSONAL MEMORY"},
    )

    assert rejected.status_code == 400
    assert accepted.status_code == 200
    assert accepted.json()["personal_claims"] == 1


def test_gap_and_external_routes_are_loopback_only_and_user_first(tmp_path):
    app, _claim = _app_with_claim(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 50000))

    gap = client.post(
        "/v1/personal-memory/understanding-gap/route",
        json={"kind": "intent_ambiguity"},
    )
    outside = client.post(
        "/v1/personal-memory/external-reflection/prepare",
        json={"query": "general habit research"},
    )

    assert gap.json()["action"] == "ask_user"
    assert outside.json()["action"] == "hold_local"

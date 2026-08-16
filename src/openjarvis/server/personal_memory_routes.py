"""Authenticated, loopback-only controls for developmental personal memory."""

from __future__ import annotations

import ipaddress
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from openjarvis.memory.inspector import PersonalMemoryInspector

router = APIRouter(prefix="/v1/personal-memory", tags=["personal-memory"])


class SuppressRequest(BaseModel):
    reason: str = "user requested"


class CorrectionRequest(BaseModel):
    replacement_text: str
    user_text: str


class ConfirmationRequest(BaseModel):
    user_text: str


class DeleteAllRequest(BaseModel):
    confirm_token: str


class GapRequest(BaseModel):
    kind: str
    user_requested_analysis: bool = False
    sensitive: bool = False


class ExternalReflectionRequest(BaseModel):
    query: str
    user_requested_analysis: bool = False
    consent_granted: bool = False
    sensitive: bool = False


class QuestionAnswerRequest(BaseModel):
    answer_text: str
    confirms: bool


def _require_local_inspector(request: Request) -> PersonalMemoryInspector:
    """Reject every personal-memory operation unless the client is loopback."""
    host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if not is_loopback:
        raise HTTPException(status_code=403, detail="Local access required")
    service = getattr(request.app.state, "personal_memory_service", None)
    if service is None:
        raise HTTPException(status_code=404, detail="Personal memory is disabled")
    return PersonalMemoryInspector(service.archive)


def _require_local_service(request: Request):
    _require_local_inspector(request)
    return request.app.state.personal_memory_service


@router.get("")
async def list_personal_memory(request: Request):
    """List safe read models without unrelated archive records."""
    return [
        asdict(item)
        for item in _require_local_inspector(request).list_subjects()
    ]


@router.get("/deletion-preview")
async def deletion_preview(request: Request):
    """Return affected row counts without exposing stored text."""
    return _require_local_inspector(request).deletion_preview()


@router.post("/understanding-gap/route")
async def route_understanding_gap(body: GapRequest, request: Request):
    """Choose whether to ask, hold locally, or offer outside knowledge."""
    try:
        result = _require_local_service(request).route_understanding_gap(
            body.kind,
            user_requested_analysis=body.user_requested_analysis,
            sensitive=body.sensitive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(result)


@router.post("/external-reflection/prepare")
async def prepare_external_reflection(
    body: ExternalReflectionRequest,
    request: Request,
):
    """Validate explicit request and consent without making a network call."""
    result = _require_local_service(request).prepare_external_reflection(
        body.query,
        user_requested_analysis=body.user_requested_analysis,
        consent_granted=body.consent_granted,
        sensitive=body.sensitive,
    )
    return asdict(result)


@router.post("/questions/{question_id}/answer")
async def answer_personal_question(
    question_id: str,
    body: QuestionAnswerRequest,
    request: Request,
):
    """Record a clarification and apply only an explicit confirmation."""
    try:
        result = _require_local_service(request).answer_personal_question(
            question_id,
            body.answer_text,
            confirms=body.confirms,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(result)


@router.get("/{subject_id}")
async def explain_personal_memory(subject_id: str, request: Request):
    """Explain one memory using only its directly linked evidence."""
    result = _require_local_inspector(request).explain(subject_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return asdict(result)


@router.post("/{subject_id}/confirm")
async def confirm_personal_memory(
    subject_id: str,
    body: ConfirmationRequest,
    request: Request,
):
    """Confirm one memory with fresh user evidence."""
    result = _require_local_inspector(request).confirm(subject_id, body.user_text)
    if result is None:
        raise HTTPException(status_code=400, detail="Memory could not be confirmed")
    return {"id": result.id, "state": result.state.value}


@router.post("/{subject_id}/correct")
async def correct_personal_memory(
    subject_id: str,
    body: CorrectionRequest,
    request: Request,
):
    """Supersede one memory with a user-confirmed correction."""
    result = _require_local_inspector(request).correct(
        subject_id,
        body.replacement_text,
        body.user_text,
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Memory could not be corrected")
    return {"id": result.id, "state": result.state.value}


@router.post("/{subject_id}/suppress")
async def suppress_personal_memory(
    subject_id: str,
    body: SuppressRequest,
    request: Request,
):
    """Hide one memory from the next response context."""
    inspector = _require_local_inspector(request)
    if not inspector.suppress(subject_id, body.reason):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "suppressed"}


@router.post("/{subject_id}/restore")
async def restore_personal_memory(subject_id: str, request: Request):
    """Restore one suppressed memory."""
    if not _require_local_inspector(request).restore(subject_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "active"}


@router.delete("/{subject_id}")
async def delete_personal_memory(
    subject_id: str,
    request: Request,
    include_raw_evidence: bool = False,
):
    """Delete one subject and optionally its linked raw evidence."""
    try:
        removed = _require_local_inspector(request).delete_subject(
            subject_id,
            include_raw_evidence=include_raw_evidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not any(removed.values()):
        raise HTTPException(status_code=404, detail="Memory not found")
    return removed


@router.delete("")
async def delete_all_personal_memory(body: DeleteAllRequest, request: Request):
    """Delete every personal record after exact-token confirmation."""
    try:
        return _require_local_inspector(request).delete_all_personal_memory(
            body.confirm_token
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]

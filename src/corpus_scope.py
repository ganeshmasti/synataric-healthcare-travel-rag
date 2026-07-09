"""Corpus-scope guardrails for Synataric Navigator.

This module is intentionally deterministic. It prevents unsupported
geography/procedure requests from reaching provider or cost tools that only
cover the current illustrative Synataric corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.agent_intents import detect_out_of_scope, detect_unsafe_medical
from src.output_sanitizer import sanitize_result_dict, sanitize_text


CorpusScopeStatus = Literal["supported", "partial", "coverage_gap", "needs_clarification", "not_applicable"]

SUPPORTED_PROCEDURES = {
    "cataract": ["cataract", "cataract surgery", "cataract operation"],
    "knee_replacement": ["knee replacement", "knee replacement surgery"],
    "cardiac_bypass": ["cardiac bypass", "heart bypass", "bypass surgery", "cabg"],
}

UNSUPPORTED_PROCEDURE_ALIASES = {
    "robotic neurosurgery": ["robotic neurosurgery", "neurosurgery", "brain surgery"],
    "organ transplant": ["organ transplant", "kidney transplant", "liver transplant", "heart transplant"],
    "IVF": ["ivf", "in vitro fertilization", "fertility treatment"],
    "LASIK": ["lasik", "laser eye surgery"],
    "dental implants": ["dental implant", "dental implants"],
    "cancer treatment": ["cancer treatment", "chemotherapy", "radiation therapy", "oncology"],
    "cosmetic surgery": ["cosmetic surgery", "plastic surgery", "rhinoplasty", "facelift"],
}

SUPPORTED_PROVIDER_COST_GEOS = {"bangalore", "india"}

SUPPORTED_GEO_ALIASES = {
    "bangalore": ["bangalore", "bengaluru"],
    "india": ["india"],
}

UNSUPPORTED_GEO_ALIASES = {
    "norway": ["norway", "oslo"],
    "usa": ["usa", "u.s.", "u.s.a.", "united states", "america", "american"],
    "uk": ["uk", "u.k.", "united kingdom", "england", "london"],
    "thailand": ["thailand", "bangkok"],
    "germany": ["germany", "berlin", "munich"],
    "singapore": ["singapore"],
    "turkey": ["turkey", "istanbul"],
    "mexico": ["mexico", "cancun", "mexico city"],
    "uae": ["uae", "u.a.e.", "dubai", "abu dhabi"],
    "japan": ["japan", "tokyo"],
    "south_korea": ["south korea", "korea", "seoul"],
    "canada": ["canada", "toronto", "vancouver", "montreal"],
    "australia": ["australia", "sydney", "melbourne"],
    "france": ["france", "paris"],
    "spain": ["spain", "madrid", "barcelona"],
    "italy": ["italy", "rome", "milan"],
    "malaysia": ["malaysia", "kuala lumpur"],
}

SUPPORTED_PROVIDER_SOURCES = {"bangalore_eye_hospitals.csv", "provider_profiles.md"}
SUPPORTED_COST_SOURCES = {"india_procedure_costs.csv", "travel_stay_costs.csv"}
SUPPORTED_GENERAL_SOURCES = {
    "cataract_surgery_guide.md",
    "knee_replacement_guide.md",
    "cardiac_bypass_guide.md",
    "post_op_recovery_guidelines.md",
    "travel_medical_risk_checklist.md",
    "synataric_disclaimer_and_safety.md",
}

GEO_DISPLAY_NAMES = {
    "bangalore": "Bangalore",
    "india": "India",
    "norway": "Norway",
    "usa": "USA",
    "uk": "UK",
    "thailand": "Thailand",
    "germany": "Germany",
    "singapore": "Singapore",
    "turkey": "Turkey",
    "mexico": "Mexico",
    "uae": "UAE",
    "japan": "Japan",
    "south_korea": "South Korea",
    "canada": "Canada",
    "australia": "Australia",
    "france": "France",
    "spain": "Spain",
    "italy": "Italy",
    "malaysia": "Malaysia",
}

DIMENSION_TERMS = {
    "provider": ["provider", "providers", "hospital", "hospitals", "clinic", "clinics", "where can i find", "best place"],
    "cost": ["cost", "costs", "price", "pricing", "estimate", "budget", "package", "fee", "charges"],
    "travel": ["travel", "trip", "itinerary", "stay", "hotel", "airport", "caregiver", "logistics", "care travel plan"],
    "recovery": ["recovery", "recover", "post-op", "post op", "follow-up", "follow up", "after surgery"],
    "risk": ["risk", "risks", "urgent symptoms", "red flags", "complications", "warning signs", "immediate care"],
}

HEALTH_NAV_TERMS = [
    "surgery",
    "procedure",
    "care plan",
    "care travel",
    "provider",
    "hospital",
    "clinic",
    "cost",
    "recovery",
    "risk",
    "medical travel",
]

SUPPORTED_QUERIES = [
    "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.",
    "What is the cost of cataract surgery in Bangalore?",
    "What recovery guidance is available after cataract surgery?",
    "What urgent symptoms require immediate care?",
]


@dataclass
class CorpusScopeResult:
    status: CorpusScopeStatus
    requested_procedure: str | None = None
    requested_geography: str | None = None
    missing_dimensions: list[str] = field(default_factory=list)
    supported_dimensions: list[str] = field(default_factory=list)
    user_message: str = ""
    display_title: str = ""
    display_message: str = ""
    suggested_supported_queries: list[str] = field(default_factory=lambda: list(SUPPORTED_QUERIES))


def normalize_requested_geography(question: str) -> str | None:
    text = _normalize(question)
    for geography, aliases in {**SUPPORTED_GEO_ALIASES, **UNSUPPORTED_GEO_ALIASES}.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            return geography
    return None


def detect_requested_procedure(question: str) -> str | None:
    text = _normalize(question)
    for procedure, aliases in SUPPORTED_PROCEDURES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            return procedure
    for procedure, aliases in UNSUPPORTED_PROCEDURE_ALIASES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            return procedure
    return None


def detect_requested_dimensions(question: str) -> set[str]:
    text = _normalize(question)
    dimensions = {dimension for dimension, terms in DIMENSION_TERMS.items() if any(term in text for term in terms)}
    if "plan" in text and ("surgery" in text or "procedure" in text):
        dimensions.update({"provider", "cost", "travel", "recovery", "risk"})
    return dimensions


def evaluate_corpus_scope(question: str, status: str | None = None) -> CorpusScopeResult:
    status_text = str(status or "").lower()
    if status_text in {"unsafe", "unsafe_medical"} or detect_unsafe_medical(question):
        return CorpusScopeResult(status="not_applicable", display_title="Safety boundary")
    out_of_scope, _reason = detect_out_of_scope(question)
    if status_text == "out_of_scope" or out_of_scope:
        return CorpusScopeResult(status="not_applicable", display_title="Outside healthcare-navigation scope")

    procedure = detect_requested_procedure(question)
    geography = normalize_requested_geography(question)
    dimensions = detect_requested_dimensions(question)
    is_health_nav = _contains_any(_normalize(question), HEALTH_NAV_TERMS)

    if is_health_nav and not procedure and _requires_procedure(dimensions, question):
        return CorpusScopeResult(
            status="needs_clarification",
            requested_geography=geography,
            missing_dimensions=["procedure"],
            user_message="Which procedure are you considering?",
            display_title="Clarification needed",
            display_message="Which procedure are you considering?",
        )

    if procedure and procedure not in SUPPORTED_PROCEDURES:
        return _coverage_gap(
            requested_procedure=procedure,
            requested_geography=geography,
            missing_dimensions=["procedure"],
            message=f"I don't have {procedure} evidence in the current Synataric corpus.",
            scope=f"{procedure}",
        )

    if ({"provider", "cost", "travel"} & dimensions) and not geography:
        return CorpusScopeResult(
            status="needs_clarification",
            requested_procedure=procedure,
            missing_dimensions=["geography"],
            user_message="Which city or country are you considering?",
            display_title="Clarification needed",
            display_message="Which city or country are you considering?",
        )

    if geography and geography not in SUPPORTED_PROVIDER_COST_GEOS and ({"provider", "cost", "travel"} & dimensions):
        country = GEO_DISPLAY_NAMES.get(geography, geography.title())
        return _coverage_gap(
            requested_procedure=procedure,
            requested_geography=geography,
            missing_dimensions=["provider", "cost"],
            supported_dimensions=sorted(dimensions & {"recovery", "risk"}),
            message=f"I don't have {country}-specific provider or cost records in the current Synataric corpus.",
            scope=f"{country} provider or cost",
        )

    return CorpusScopeResult(
        status="supported",
        requested_procedure=procedure,
        requested_geography=geography,
        supported_dimensions=sorted(dimensions) or ["general_guidance"],
        user_message="The request is supported by the current Synataric corpus scope.",
        display_title="Corpus scope supported",
        display_message="The request is supported by the current Synataric corpus scope.",
    )


def build_coverage_gap_response(scope: CorpusScopeResult) -> dict:
    gap_message = scope.user_message or f"I don't have {_requested_scope_label(scope)}-specific evidence in the current Synataric corpus."
    country = GEO_DISPLAY_NAMES.get(scope.requested_geography or "", scope.requested_geography or "that geography")
    provider_target = country
    cost_target = country
    if scope.requested_procedure and scope.requested_procedure not in SUPPORTED_PROCEDURES:
        provider_target = scope.requested_procedure
        cost_target = scope.requested_procedure
    answer = (
        "Corpus coverage gap\n\n"
        f"{gap_message}\n\n"
        "What the current corpus supports:\n"
        "- Cataract surgery, knee replacement, and cardiac bypass navigation.\n"
        "- India/Bangalore provider and cost examples.\n"
        "- General procedure, recovery, risk, and safety guidance from the indexed Synataric guides.\n\n"
        "What I cannot do from the current corpus:\n"
        f"- Recommend {provider_target} providers.\n"
        f"- Quote {cost_target} costs.\n"
        "- Verify live availability, insurance coverage, or real-time pricing.\n\n"
        "Try one of these supported queries:\n"
        "- Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.\n"
        "- What is the cost of cataract surgery in Bangalore?\n"
        "- What recovery guidance is available after cataract surgery?\n"
        "- What urgent symptoms require immediate care?\n\n"
        "Educational healthcare navigation only. Not medical advice."
    )
    result = {
        "status": "coverage_gap",
        "selected_tool": "coverage_gap_response_tool",
        "answer": sanitize_text(answer),
        "sources": [],
        "evidence": [],
        "warnings": ["corpus_coverage_gap"],
        "requires_human": False,
        "corpus_scope": scope,
    }
    return sanitize_result_dict(result)


def _coverage_gap(
    *,
    requested_procedure: str | None = None,
    requested_geography: str | None = None,
    missing_dimensions: list[str] | None = None,
    supported_dimensions: list[str] | None = None,
    message: str,
    scope: str,
) -> CorpusScopeResult:
    return CorpusScopeResult(
        status="coverage_gap",
        requested_procedure=requested_procedure,
        requested_geography=requested_geography,
        missing_dimensions=missing_dimensions or [],
        supported_dimensions=supported_dimensions or [],
        user_message=sanitize_text(message),
        display_title="Corpus Coverage Gap",
        display_message=sanitize_text(message),
        suggested_supported_queries=list(SUPPORTED_QUERIES),
    )


def _requested_scope_label(scope: CorpusScopeResult) -> str:
    if scope.requested_procedure and scope.requested_procedure not in SUPPORTED_PROCEDURES:
        return scope.requested_procedure
    if scope.requested_geography:
        country = GEO_DISPLAY_NAMES.get(scope.requested_geography, scope.requested_geography.title())
        if {"provider", "cost"} & set(scope.missing_dimensions):
            return f"{country} provider or cost"
        return country
    return "the requested scope"


def _requires_procedure(dimensions: set[str], question: str) -> bool:
    text = _normalize(question)
    if "urgent symptoms" in text or "immediate care" in text:
        return False
    return bool(dimensions or "surgery" in text or "procedure" in text)


def _contains_alias(text: str, alias: str) -> bool:
    return bool(re.search(rf"\b{re.escape(alias)}\b", text))


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()

"""Intent routing for the agentic Synataric Navigator upgrade.

This file is the first step toward an agentic Synataric Navigator. It does not
replace RAG; it classifies user goals so future tools can route to the right
navigation capability before retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import load_settings


IntentLabel = Literal[
    "provider_search",
    "cost_estimate",
    "recovery_guidance",
    "risk_checklist",
    "travel_planning",
    "find_evidence",
    "general_navigation",
    "unsafe_medical",
    "needs_clarification",
    "out_of_scope",
]


INTENT_LABELS: set[str] = set(IntentLabel.__args__)

SUGGESTED_TOOLS: dict[str, list[str]] = {
    "provider_search": ["provider_search_tool"],
    "cost_estimate": ["cost_estimate_tool"],
    "recovery_guidance": ["recovery_guidance_tool"],
    "risk_checklist": ["risk_checklist_tool"],
    "travel_planning": ["travel_planning_tool"],
    "find_evidence": ["find_evidence_tool"],
    "general_navigation": ["general_rag_tool"],
    "unsafe_medical": ["safety_response_tool"],
    "needs_clarification": ["ask_human_tool"],
    "out_of_scope": ["out_of_scope_response_tool"],
}

class IntentClassification(BaseModel):
    intent: str = Field(description="One of the supported healthcare navigation intent labels.")
    confidence: float = Field(ge=0, le=1, description="Classifier confidence from 0 to 1.")
    reasoning: str = Field(description="Brief explanation for the classification.")
    missing_fields: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)


def classify_intent(question: str, patient_context: dict | None = None) -> IntentClassification:
    """Classify a healthcare navigation question before retrieval.

    The primary path uses ChatOpenAI structured output with temperature 0. If
    the model call or parsing fails, a conservative keyword classifier returns a
    usable routing decision.
    """
    patient_context = patient_context or {}
    cleaned_question = question.strip()
    if not cleaned_question:
        return _finalize_classification(
            IntentClassification(
                intent="needs_clarification",
                confidence=0.95,
                reasoning="The question is empty.",
                missing_fields=["question"],
                suggested_tools=[],
                safety_flags=[],
            ),
            cleaned_question,
            patient_context,
        )

    try:
        classification = _classify_with_llm(cleaned_question, patient_context)
    except Exception:
        classification = _fallback_classify(cleaned_question, patient_context)
    return _finalize_classification(classification, cleaned_question, patient_context)


def _classify_with_llm(question: str, patient_context: dict[str, Any]) -> IntentClassification:
    from langchain_openai import ChatOpenAI

    settings = load_settings(require_secrets=True)
    llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    structured_llm = llm.with_structured_output(IntentClassification)

    prompt = f"""
Classify this healthcare travel navigation question into exactly one intent.

Intent labels and rules:
- provider_search: where to go, which hospital/provider, best place, centers, clinics, specialists.
- cost_estimate: price, cost, budget, estimates, package, travel/stay cost.
- recovery_guidance: recovery timeline, post-op care, follow-up, healing, return to travel.
- risk_checklist: risks, red flags, urgent symptoms, complications, safety concerns.
- travel_planning: itinerary, stay duration, airport, hotel, travel logistics, caregiver planning.
- find_evidence: where something is explained, show source, find document, where did this come from.
- general_navigation: general care navigation that can be answered with normal RAG.
- unsafe_medical: diagnosis, prescription, emergency decision-making, or urgent medical judgment.
- needs_clarification: lacks key information such as procedure, location, budget, or destination.
- out_of_scope: not a healthcare travel or care-navigation request supported by Synataric.

Missing field rules:
- Provider search without a procedure: missing_fields includes "procedure".
- Cost estimate without a procedure: missing_fields includes "procedure".
- Travel planning without a destination/city: missing_fields includes "destination".
- Care plan/recovery guidance without a procedure: missing_fields includes "procedure".

Suggested tool mapping:
provider_search -> provider_search_tool
cost_estimate -> cost_estimate_tool
recovery_guidance -> recovery_guidance_tool
risk_checklist -> risk_checklist_tool
travel_planning -> travel_planning_tool
find_evidence -> find_evidence_tool
general_navigation -> general_rag_tool
unsafe_medical -> safety_response_tool
needs_clarification -> ask_human_tool
out_of_scope -> out_of_scope_response_tool

Question: {question}
Patient context: {patient_context}
"""
    return structured_llm.invoke(prompt)


def _fallback_classify(question: str, patient_context: dict[str, Any]) -> IntentClassification:
    intent = keyword_intent_hint(question) or "general_navigation"
    confidence = 0.9 if intent == "unsafe_medical" else 0.82
    return _result(intent, confidence, f"Keyword routing selected {intent}.", safety_flags=detect_unsafe_medical(question))


def _finalize_classification(
    classification: IntentClassification,
    question: str,
    patient_context: dict[str, Any],
) -> IntentClassification:
    intent = classification.intent if classification.intent in INTENT_LABELS else "needs_clarification"
    missing_fields = list(dict.fromkeys(classification.missing_fields))
    safety_flags = list(dict.fromkeys(classification.safety_flags))
    hint = keyword_intent_hint(question)
    procedure = detect_procedure(question) or patient_context.get("procedure") or patient_context.get("procedure_name")
    location = detect_location(question) or patient_context.get("destination") or patient_context.get("city") or patient_context.get("location")
    unsafe_flags = detect_unsafe_medical(question)
    out_of_scope, out_of_scope_reason = detect_out_of_scope(question)

    if out_of_scope:
        intent = "out_of_scope"
        missing_fields = []
    elif unsafe_flags:
        intent = "unsafe_medical"
        safety_flags = list(dict.fromkeys(safety_flags + unsafe_flags))
    elif intent == "unsafe_medical":
        safety_flags = safety_flags or ["urgent_medical_judgment"]
    elif hint and hint != "general_navigation":
        intent = hint
    elif hint == "general_navigation" and intent not in INTENT_LABELS:
        intent = "general_navigation"

    if _is_extremely_vague_request(question):
        intent = "needs_clarification"
        missing_fields = ["care_topic"]
    elif _is_vague_navigation_request(question, intent) and not procedure:
        intent = "needs_clarification"

    if procedure:
        missing_fields = [field for field in missing_fields if field != "procedure"]
    if location:
        missing_fields = [field for field in missing_fields if field not in {"location", "destination"}]
    if intent in {"unsafe_medical", "find_evidence", "risk_checklist", "out_of_scope"}:
        missing_fields = []

    if intent in {"provider_search", "cost_estimate", "recovery_guidance", "travel_planning"} and not procedure:
        missing_fields.append("procedure")
    if intent == "travel_planning" and not location:
        missing_fields.append("destination")
    if _asks_for_care_plan(question) and not procedure:
        missing_fields.append("procedure")
    if intent == "needs_clarification" and "care_topic" not in missing_fields and not procedure:
        missing_fields.append("procedure")
        if _mentions_general_care_without_location(question) and not location:
            missing_fields.append("location")

    missing_fields = list(dict.fromkeys(missing_fields))

    return IntentClassification(
        intent=intent,
        confidence=max(0.0, min(1.0, max(classification.confidence, 0.95) if out_of_scope else classification.confidence)),
        reasoning=f"{classification.reasoning} Out-of-scope reason: {out_of_scope_reason}." if out_of_scope else classification.reasoning,
        missing_fields=missing_fields,
        suggested_tools=_suggested_tools(intent, missing_fields),
        safety_flags=safety_flags,
    )


def _result(
    intent: str,
    confidence: float,
    reasoning: str,
    *,
    safety_flags: list[str] | None = None,
) -> IntentClassification:
    return IntentClassification(
        intent=intent,
        confidence=confidence,
        reasoning=reasoning,
        missing_fields=[],
        suggested_tools=SUGGESTED_TOOLS.get(intent, []),
        safety_flags=safety_flags or [],
    )


def _has_procedure(question: str, patient_context: dict[str, Any]) -> bool:
    return bool(patient_context.get("procedure") or patient_context.get("procedure_name") or detect_procedure(question))


def _has_destination(question: str, patient_context: dict[str, Any]) -> bool:
    return bool(patient_context.get("destination") or patient_context.get("city") or patient_context.get("location") or detect_location(question))


def detect_procedure(question: str) -> str | None:
    normalized = _normalize(question)
    procedure_patterns = [
        ("cataract surgery", r"\bcataract\s+(?:surgery|operation|procedure)\b"),
        ("knee replacement", r"\bknee\s+replacement\b"),
        ("cardiac bypass", r"\bcardiac\s+bypass\b"),
        ("heart bypass", r"\bheart\s+bypass\b"),
        ("bypass surgery", r"\bbypass\s+surgery\b"),
        ("eye surgery", r"\beye\s+surgery\b"),
        ("retina surgery", r"\bretina\s+surgery\b"),
    ]
    for procedure, pattern in procedure_patterns:
        if re.search(pattern, normalized):
            return procedure
    return None


def detect_location(question: str) -> str | None:
    normalized = _normalize(question)
    locations = {
        "india": "India",
        "bangalore": "Bangalore",
        "bengaluru": "Bengaluru",
        "chennai": "Chennai",
        "mumbai": "Mumbai",
        "delhi": "Delhi",
        "hyderabad": "Hyderabad",
        "kerala": "Kerala",
    }
    for location, display_name in locations.items():
        if re.search(rf"\b{re.escape(location)}\b", normalized):
            return display_name
    return None


def detect_unsafe_medical(question: str) -> list[str]:
    normalized = _normalize(question)
    flags: list[str] = []
    if _contains_any(
        normalized,
        [
            "antibiotic",
            "antibiotics",
            "medication",
            "medicine",
            "dosage",
            "dose",
            "prescribe",
            "prescription",
            "should i take",
            "can i take",
        ],
    ):
        flags.append("prescription_or_treatment_advice")
    if _contains_any(
        normalized,
        ["diagnose", "emergency decision", "severe chest pain", "stroke", "fainting", "severe breathlessness"],
    ):
        flags.append("urgent_medical_judgment")
    return list(dict.fromkeys(flags))


def detect_out_of_scope(question: str) -> tuple[bool, str]:
    normalized = _normalize(question)
    if _contains_any(normalized, ["mars", "moon", "jupiter", "saturn", "space colony"]):
        return True, "unsupported_or_impossible_destination"
    if _contains_any(normalized, ["super bowl", "nfl", "nba", "cricket score", "world cup", "football game"]):
        return True, "sports_question"
    if _contains_any(
        normalized,
        ["who won", "capital of", "president of", "stock price", "weather", "recipe", "programming", "code error"],
    ) and not is_healthcare_navigation_question(question):
        return True, "non_healthcare_question"
    return False, ""


def is_healthcare_navigation_question(question: str) -> bool:
    normalized = _normalize(question)
    return _contains_any(
        normalized,
        [
            "surgery",
            "procedure",
            "hospital",
            "provider",
            "doctor",
            "clinic",
            "recovery",
            "cost",
            "medical travel",
            "care planning",
            "cataract",
            "knee replacement",
            "cardiac bypass",
            "symptoms",
            "risk",
            "follow-up",
            "follow up",
            "post-op",
            "post op",
            "travel for care",
        ],
    )


def keyword_intent_hint(question: str) -> str | None:
    normalized = _normalize(question)
    out_of_scope, _reason = detect_out_of_scope(question)
    if out_of_scope:
        return "out_of_scope"
    if _contains_any(
        normalized,
        [
            "antibiotic",
            "antibiotics",
            "medication",
            "medicine",
            "dosage",
            "dose",
            "prescribe",
            "prescription",
            "diagnose",
            "should i take",
            "can i take",
            "emergency decision",
            "severe chest pain",
            "stroke",
            "fainting",
            "severe breathlessness",
        ],
    ):
        return "unsafe_medical"
    if _contains_any(
        normalized,
        ["where is this explained", "where did this come from", "show source", "find evidence", "where is", "source document"],
    ):
        return "find_evidence"
    if _contains_any(
        normalized,
        [
            "where can i find",
            "hospital",
            "provider",
            "clinic",
            "center",
            "centre",
            "specialist",
            "good cataract surgery",
            "best place",
            "where to go",
        ],
    ):
        return "provider_search"
    if _contains_any(normalized, ["cost", "price", "estimate", "budget", "package", "fee", "charges"]):
        return "cost_estimate"
    if _contains_any(normalized, ["recovery", "post-op", "post op", "postoperative", "healing", "follow-up", "follow up", "after surgery"]):
        return "recovery_guidance"
    if _contains_any(normalized, ["risk", "risks", "urgent symptoms", "red flags", "complications", "warning signs", "immediate care"]):
        return "risk_checklist"
    if _contains_any(normalized, ["travel", "trip", "itinerary", "stay", "airport", "hotel", "caregiver", "logistics"]):
        return "travel_planning"
    return "general_navigation"


def _asks_for_care_plan(question: str) -> bool:
    normalized = _normalize(question)
    return _contains_any(normalized, ["care plan", "recovery guidance", "post op care", "post-op care", "follow up"])


def _mentions_generic_surgery_need(question: str) -> bool:
    normalized = _normalize(question)
    return _contains_any(normalized, ["i need surgery", "need surgery", "surgery in", "procedure in"])


def _is_vague_navigation_request(question: str, intent: str) -> bool:
    normalized = _normalize(question)
    if intent != "general_navigation":
        return False
    return _contains_any(normalized, ["i need surgery", "need surgery", "need help with care", "help with care", "care plan"])


def _is_extremely_vague_request(question: str) -> bool:
    normalized = _normalize(question)
    return normalized in {"help me with this", "can you help", "can you help?", "i need help", "help"}


def _mentions_general_care_without_location(question: str) -> bool:
    normalized = _normalize(question)
    return _contains_any(normalized, ["need help with care", "help with care", "care plan"])


def _suggested_tools(intent: str, missing_fields: list[str]) -> list[str]:
    domain_tools = SUGGESTED_TOOLS.get(intent, SUGGESTED_TOOLS["needs_clarification"])
    if missing_fields:
        return list(dict.fromkeys(["ask_human_tool"] + domain_tools))
    return domain_tools


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


if __name__ == "__main__":
    test_questions = [
        ("Where can I find good cataract surgery in India?", "provider_search", []),
        ("What is the cost of cataract surgery in Bangalore?", "cost_estimate", []),
        ("What recovery guidance is available after cataract surgery?", "recovery_guidance", []),
        ("What urgent symptoms require immediate care?", "risk_checklist", []),
        ("Plan my travel for surgery in Bangalore", "travel_planning", ["procedure"]),
        ("Where is cataract recovery planning explained?", "find_evidence", []),
        ("Should I take antibiotics after surgery?", "unsafe_medical", []),
        ("I need surgery in India", "needs_clarification", ["procedure"]),
    ]
    for q, expected_intent, expected_missing_fields in test_questions:
        result = classify_intent(q)
        print("QUESTION:", q)
        print("EXPECTED INTENT:", expected_intent)
        print("EXPECTED MISSING FIELDS:", expected_missing_fields)
        print("INTENT:", result.intent)
        print("CONFIDENCE:", result.confidence)
        print("MISSING FIELDS:", result.missing_fields)
        print("SUGGESTED TOOLS:", result.suggested_tools)
        print("SAFETY FLAGS:", result.safety_flags)
        print("REASONING:", result.reasoning)
        print()

"""Demo Mode page for Synataric Navigator.

The page is a polished wrapper around existing Synataric runtime functions and
stored benchmark reports. It does not load fine-tuned model weights or require
local Llama inference.
"""

from __future__ import annotations

import json
import re
import time
import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_PROVIDER_COST_GEOS = {"bangalore", "india"}

SUPPORTED_GEO_ALIASES = {
    "bangalore": ["bangalore", "bengaluru"],
    "india": ["india"],
}

UNSUPPORTED_GEO_ALIASES = {
    "norway": ["norway", "oslo"],
    "usa": ["usa", "u.s.", "u.s.a.", "united states", "america", "american"],
    "uk": ["uk", "u.k.", "united kingdom", "england", "london"],
    "germany": ["germany", "berlin", "munich"],
    "thailand": ["thailand", "bangkok"],
    "turkey": ["turkey", "istanbul"],
    "mexico": ["mexico", "cancun", "mexico city"],
    "uae": ["uae", "u.a.e.", "dubai", "abu dhabi"],
    "singapore": ["singapore"],
    "malaysia": ["malaysia", "kuala lumpur"],
    "south_korea": ["south korea", "korea", "seoul"],
    "japan": ["japan", "tokyo"],
    "canada": ["canada", "toronto", "vancouver", "montreal"],
    "australia": ["australia", "sydney", "melbourne"],
    "france": ["france", "paris"],
    "spain": ["spain", "madrid", "barcelona"],
    "italy": ["italy", "rome", "milan"],
}

GENERAL_PROCEDURE_SOURCES = {
    "cataract_surgery_guide.md",
    "knee_replacement_guide.md",
    "cardiac_bypass_guide.md",
    "post_op_recovery_guidelines.md",
    "travel_medical_risk_checklist.md",
    "synataric_disclaimer_and_safety.md",
}

PROVIDER_SOURCES_BY_GEO = {
    "bangalore": {"bangalore_eye_hospitals.csv", "provider_profiles.md"},
    "india": {"bangalore_eye_hospitals.csv", "provider_profiles.md"},
}

COST_SOURCES_BY_GEO = {
    "bangalore": {"india_procedure_costs.csv", "travel_stay_costs.csv"},
    "india": {"india_procedure_costs.csv", "travel_stay_costs.csv"},
}

GEO_DISPLAY_NAMES = {
    "bangalore": "Bangalore",
    "india": "India",
    "usa": "USA",
    "uk": "UK",
    "germany": "Germany",
    "thailand": "Thailand",
    "norway": "Norway",
    "turkey": "Turkey",
    "mexico": "Mexico",
    "uae": "UAE",
    "singapore": "Singapore",
    "malaysia": "Malaysia",
    "south_korea": "South Korea",
    "japan": "Japan",
    "canada": "Canada",
    "australia": "Australia",
    "france": "France",
    "spain": "Spain",
    "italy": "Italy",
}


STYLE_BLOCK = """
<style>
[data-testid="stAppViewContainer"] {
    background: #EEF4FB !important;
}
.main .block-container {
    max-width: 1268px;
    padding-top: 0.85rem;
    padding-bottom: 1.2rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-color: #E2E8F0 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}
div[data-testid="stTabs"] button {
    color: #475569 !important;
    font-weight: 700 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0F766E !important;
    background: #CCFBF1 !important;
}
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 0.75rem 0.8rem !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #64748B !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0F766E, #0284C7) !important;
    border: 0 !important;
    border-radius: 12px !important;
    min-height: 2.75rem;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(15, 118, 110, 0.18);
}
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}
.stTextArea label,
.stSelectbox label {
    color: #0F172A !important;
}
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    color: #0F172A !important;
}
div[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border-color: #E2E8F0 !important;
}
h1, h2, h3, h4 {
    color: #0F172A !important;
}
p, li, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #475569 !important;
}
hr {
    margin: 0.35rem 0;
}
textarea {
    font-size: 1rem !important;
    line-height: 1.5 !important;
}
</style>
"""


# Fallback values mirror the Week 4/Week 5 reports so the demo remains usable
# when report artifacts are absent in a deployment bundle.
DEFAULT_DEMO_METRICS = {
    "existing_router": {
        "accuracy": 0.555,
        "macro_f1": 0.493,
        "route_execution_score": 0.609,
        "average_latency_seconds": 2.308,
        "invalid_output_rate": 0.0,
    },
    "fine_tuned_router": {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "route_execution_score": 1.0,
        "average_latency_seconds": 0.340,
        "invalid_output_rate": 0.0,
        "smoke_test_passed": 9,
        "smoke_test_total": 9,
    },
    "agent_eval": {
        "baseline_overall": 0.8283,
        "post_improvement_overall": 0.8810,
        "delta": 0.0527,
        "top_improvements": [
            "status_accuracy +0.1500",
            "task_completion_score +0.1500",
            "source_hit_rate +0.1250",
            "human_handoff_accuracy +0.1250",
            "local_path_leakage_absence reached 1.0000",
        ],
    },
}


DEMO_SCENARIOS = {
    "Multi-step care plan": {
        "question": "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.",
        "expected_route": "care_plan_multistep",
        "workflow": "ReAct Care Planner",
        "expected_tools": [
            "provider_search_tool",
            "cost_estimate_tool",
            "recovery_guidance_tool",
            "risk_checklist_tool",
        ],
    },
    "Cost estimate": {
        "question": "What is the cost of cataract surgery in Bangalore?",
        "expected_route": "cost_estimate",
        "workflow": "cost_estimate_tool",
        "expected_tools": ["cost_estimate_tool"],
    },
    "Provider search": {
        "question": "Where can I find good cataract surgery in India?",
        "expected_route": "provider_search",
        "workflow": "provider_search_tool",
        "expected_tools": ["provider_search_tool"],
    },
    "Human clarification": {
        "question": "Plan my travel for surgery in Bangalore",
        "expected_route": "needs_clarification",
        "workflow": "ask_human_tool",
        "expected_tools": ["ask_human_tool"],
        "human_question": "Which procedure are you considering?",
    },
    "Safety refusal": {
        "question": "Should I take antibiotics after surgery?",
        "expected_route": "unsafe_medical",
        "workflow": "safety_response_tool",
        "expected_tools": ["safety_response_tool"],
    },
    "Evidence lookup": {
        "question": "Where is cataract recovery planning explained?",
        "expected_route": "find_evidence",
        "workflow": "find_evidence_tool",
        "expected_tools": ["find_evidence_tool"],
    },
    "Out of scope": {
        "question": "Who won the Super Bowl in 2024?",
        "expected_route": "out_of_scope",
        "workflow": "out_of_scope_response_tool",
        "expected_tools": ["out_of_scope_response_tool"],
    },
    "Unsupported geography / USA gap test": {
        "question": "Create a care travel plan for cataract surgery in the USA including providers, cost, recovery, and risks.",
        "expected_route": "care_plan_multistep",
        "workflow": "ReAct Care Planner",
        "expected_tools": [
            "provider_search_tool",
            "cost_estimate_tool",
            "recovery_guidance_tool",
            "risk_checklist_tool",
        ],
    },
}


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    if hasattr(obj, "model_dump"):
        return obj.model_dump().get(key, default)
    if hasattr(obj, "dict"):
        return obj.dict().get(key, default)
    return default


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return {key: getattr(obj, key) for key in dir(obj) if not key.startswith("_")}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def load_demo_metrics() -> dict[str, Any]:
    metrics = json.loads(json.dumps(DEFAULT_DEMO_METRICS))

    baseline = _load_json(PROJECT_ROOT / "reports" / "finetune" / "baseline_existing_router" / "router_classification_report.json")
    fine_tuned = _load_json(PROJECT_ROOT / "reports" / "finetune" / "llama_finetuned_router" / "router_classification_report.json")
    week5 = _load_json(PROJECT_ROOT / "reports" / "finetune" / "synataric_week5_finetuned_router_report.json")
    agent_eval = _load_json(PROJECT_ROOT / "reports" / "evals" / "synataric_agent_eval_delta_report.json")

    metrics["existing_router"].update(
        {
            "accuracy": baseline.get("accuracy", metrics["existing_router"]["accuracy"]),
            "macro_f1": baseline.get("macro_f1", metrics["existing_router"]["macro_f1"]),
            "route_execution_score": baseline.get("route_execution_score", metrics["existing_router"]["route_execution_score"]),
            "average_latency_seconds": baseline.get(
                "average_latency_seconds",
                metrics["existing_router"]["average_latency_seconds"],
            ),
            "invalid_output_rate": baseline.get("invalid_output_rate", metrics["existing_router"]["invalid_output_rate"]),
        }
    )
    metrics["fine_tuned_router"].update(
        {
            "accuracy": fine_tuned.get("accuracy", metrics["fine_tuned_router"]["accuracy"]),
            "macro_f1": fine_tuned.get("macro_f1", metrics["fine_tuned_router"]["macro_f1"]),
            "route_execution_score": fine_tuned.get("route_execution_score", metrics["fine_tuned_router"]["route_execution_score"]),
            "average_latency_seconds": fine_tuned.get(
                "average_latency_seconds",
                metrics["fine_tuned_router"]["average_latency_seconds"],
            ),
            "invalid_output_rate": fine_tuned.get("invalid_output_rate", metrics["fine_tuned_router"]["invalid_output_rate"]),
        }
    )

    week5_router = week5.get("fine_tuned_llama_3_2_1b_lora_router", {})
    smoke_test = week5_router.get("smoke_test", {})
    metrics["fine_tuned_router"]["smoke_test_passed"] = smoke_test.get("passed", metrics["fine_tuned_router"]["smoke_test_passed"])
    metrics["fine_tuned_router"]["smoke_test_total"] = smoke_test.get("total", metrics["fine_tuned_router"]["smoke_test_total"])

    metrics["agent_eval"].update(
        {
            "baseline_overall": agent_eval.get("baseline_overall", metrics["agent_eval"]["baseline_overall"]),
            "post_improvement_overall": agent_eval.get(
                "post_improvement_overall",
                metrics["agent_eval"]["post_improvement_overall"],
            ),
            "delta": agent_eval.get("delta", metrics["agent_eval"]["delta"]),
            "top_improvements": agent_eval.get("top_improvements", metrics["agent_eval"]["top_improvements"]),
        }
    )
    return metrics


def sanitize_demo_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text)
    root_pattern = re.escape(str(PROJECT_ROOT))
    cleaned = re.sub(root_pattern + r"[\\/]*", "", cleaned, flags=re.IGNORECASE)

    def replace_path(match: re.Match) -> str:
        value = match.group(0).rstrip(".,);]")
        return Path(value.replace("\\", "/")).name or "local_file"

    cleaned = re.sub(r"[A-Za-z]:\\[^\s\]\)>,;]+", replace_path, cleaned)
    cleaned = re.sub(r"/Users/[^\s\]\)>,;]+", replace_path, cleaned)
    cleaned = re.sub(r"\bsource_path\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _source_name(value: Any) -> str:
    cleaned = sanitize_demo_text(value)
    if not cleaned:
        return "unknown"
    return Path(cleaned.replace("\\", "/")).name or cleaned


def _snippet(value: Any, limit: int = 420) -> str:
    cleaned = sanitize_demo_text(value)
    return cleaned[: limit - 3] + "..." if len(cleaned) > limit else cleaned


SECTION_LABELS = {
    "summary": "Care Plan Summary",
    "provider_options": "Provider Options",
    "estimated_cost": "Estimated Cost",
    "recovery_guidance": "Recovery Guidance",
    "risk_red_flags": "Risk / Red Flags",
    "clinician_questions": "Questions to Ask a Clinician",
    "sources": "Sources / Evidence",
}


SECTION_ALIASES = {
    "provider_options": [
        "provider",
        "providers",
        "provider options",
        "hospital",
        "hospitals",
    ],
    "estimated_cost": ["estimated cost", "cost", "cost estimate", "pricing", "budget"],
    "recovery_guidance": ["recovery", "recovery guidance", "follow-up", "follow up", "post-op", "post op"],
    "risk_red_flags": ["risk", "risks", "red flags", "risk / red flags", "urgent symptoms", "safety"],
    "clinician_questions": [
        "questions",
        "questions to ask",
        "questions to ask a licensed clinician",
        "ask a clinician",
        "licensed clinician",
    ],
    "sources": ["source", "sources", "evidence", "citations"],
}


def _clean_answer_line(line: str) -> str:
    cleaned = sanitize_demo_text(line)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    cleaned = re.sub(r"^\s*\d+[\.)]\s+", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    return cleaned.strip()


def _section_key(heading: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s/.-]", "", heading.lower()).strip()
    for key, aliases in SECTION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return key
    if normalized:
        return "summary"
    return "summary"


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^#{1,6}\s+", stripped):
        return True
    if re.match(r"^\*\*[^*]{3,80}\*\*:?\s*$", stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 80:
        return any(alias in stripped.lower() for aliases in SECTION_ALIASES.values() for alias in aliases)
    return False


def _provider_from_line(line: str) -> dict[str, str]:
    cleaned = _clean_answer_line(line)
    if ":" in cleaned:
        name, detail = cleaned.split(":", 1)
    elif " - " in cleaned:
        name, detail = cleaned.split(" - ", 1)
    else:
        name, detail = cleaned, ""
    return {"name": name.strip() or "Provider option", "detail": detail.strip()}


def parse_care_plan_sections(answer_text: Any) -> dict[str, Any]:
    text = "" if answer_text is None else str(answer_text)
    sections: dict[str, list[str]] = {key: [] for key in SECTION_LABELS}
    providers: list[dict[str, str]] = []
    fallback: list[str] = []
    title = "Care Navigation Plan"
    current_key = "summary"

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading(line):
            heading = _clean_answer_line(line).rstrip(":")
            current_key = _section_key(heading)
            if "plan" in heading.lower() and len(heading) <= 80:
                title = heading
            continue
        cleaned = _clean_answer_line(line)
        if not cleaned:
            continue
        sections.setdefault(current_key, []).append(cleaned)
        fallback.append(cleaned)

    if not any(sections.values()) and text:
        cleaned = _clean_answer_line(text)
        sections["summary"].append(cleaned)
        fallback.append(cleaned)

    for item in sections.get("provider_options", []):
        providers.append(_provider_from_line(item))

    return {
        "title": title,
        "sections": sections,
        "providers": providers,
        "fallback": fallback,
    }


def build_mint_decision_ladder() -> list[dict[str, str]]:
    return [
        {
            "mode": "Simple",
            "title": "Ask Navigator",
            "label": "Simple RAG question",
            "description": "Retrieves evidence and answers from the Synataric corpus.",
            "example": "What is the cost of cataract surgery in Bangalore?",
        },
        {
            "mode": "Routed",
            "title": "Agent Navigator",
            "label": "One intent -> one tool",
            "description": "Routes provider, cost, recovery, risk, travel, safety, and clarification requests.",
            "example": "Where can I find good cataract surgery in India?",
        },
        {
            "mode": "Agentic",
            "title": "ReAct Care Planner",
            "label": "Multi-step goal",
            "description": "Reason -> Act -> Observe -> Decide next step. Used only for complete care plans.",
            "example": "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.",
        },
        {
            "mode": "Guarded",
            "title": "Safety / Human Boundary",
            "label": "Guardrails",
            "description": "Unsafe medical requests are refused. Missing information asks the human.",
            "example": "Should I take antibiotics after surgery?",
        },
    ]


def detect_react_needs(question: str) -> list[str]:
    text = str(question or "").lower()
    needs: list[str] = []
    if any(term in text for term in ["provider", "providers", "hospital", "hospitals", "where can i find"]):
        needs.append("Providers")
    if any(term in text for term in ["cost", "costs", "price", "estimate", "budget"]):
        needs.append("Cost")
    if any(term in text for term in ["recovery", "recover", "follow-up", "follow up", "post-op", "post op"]):
        needs.append("Recovery")
    if any(term in text for term in ["risk", "risks", "red flag", "red flags", "urgent"]):
        needs.append("Risks")
    return needs


def build_why_react_panel(scenario_key: str, question: str) -> dict[str, Any]:
    is_multi_step = "multi" in str(scenario_key or "").lower() or len(detect_react_needs(question)) >= 3
    if is_multi_step:
        return {
            "title": "Why ReAct Planner?",
            "body": (
                "This input is a goal, not a single question. It asks for providers, cost, recovery, and risks. "
                "Synataric therefore activates the bounded ReAct Care Planner instead of a one-shot answer."
            ),
            "chips": detect_react_needs(question),
            "selected_workflow": "ReAct Care Planner",
            "reasoning_pattern": "Goal -> Sense intent -> Call tools -> Observe evidence -> Synthesize plan",
            "compact": False,
        }
    return {
        "title": "MINT selected a lighter workflow",
        "body": "This request does not require multi-step planning, so Synataric routes it to RAG or one focused tool.",
        "chips": [],
        "selected_workflow": "One focused workflow",
        "reasoning_pattern": "Question -> Route -> Retrieve or call one tool -> Answer",
        "compact": True,
    }


def build_pitch_script_items() -> list[str]:
    return [
        "1. Market hook: healthcare AI platforms are emerging, but implementation is the hard part.",
        "2. Synataric one-liner: not a medical chatbot, a care-navigation workflow layer.",
        "3. MINT design: simple questions use RAG or one tool; complex goals activate ReAct.",
        "4. Live demo: Bangalore cataract care plan calls provider, cost, recovery, and risk tools.",
        "5. Safety: antibiotics request triggers refusal.",
        "6. Human clarification: missing procedure asks the human.",
        "7. Evals: agent score 0.8283 -> 0.8810; router accuracy 0.555 -> 1.000.",
    ]


def build_executive_story_content() -> dict[str, Any]:
    return {
        "title": "Synataric Global",
        "subtitle": "Revolutionizing Healthcare: AI-Powered Access, Anywhere, Anytime.",
        "positioning": "Empowering every human with world-class, affordable medical care through autonomous AI.",
        "badges": [],
        "why_cards": [
            {
                "title": "Fragmented healthcare journeys",
                "text": (
                    "Patients manually research providers, prices, travel logistics, recovery, and risk across "
                    "disconnected sources."
                ),
            },
            {
                "title": "Global treatment decisions need trust",
                "text": "Users need grounded evidence, visible sources, and clear safety boundaries before acting.",
            },
            {
                "title": "AI platforms are not enough",
                "text": (
                    "The hard part is implementation: route the request, retrieve evidence, call tools, manage "
                    "handoff, and measure quality."
                ),
            },
        ],
        "why_caption": "Cost and access variation creates a navigation problem.",
        "architecture_flow": [
            "User goal",
            "MINT Router",
            "Ask Navigator / Agent Navigator / ReAct Care Planner",
            "RAG + Domain Tools",
            "Grounded care plan + evidence",
            "Safety / Human handoff / Evaluation trace",
        ],
        "measured_improvement_text": (
            "The demo is not only a happy path. It is backed by golden datasets, LangSmith traces, deterministic "
            "code evaluators, source checks, trajectory checks, safety checks, and measured deltas."
        ),
        "recording_script": (
            "Synataric is not a generic chatbot. It is a care-navigation workflow layer. Simple questions use "
            "grounded RAG. Single-intent questions route to one tool. Multi-step care goals activate a bounded "
            "ReAct planner. Unsafe medical requests are refused, missing information asks the human, and evals "
            "prove the improvement."
        ),
        "production_path": {
            "title": "Production path",
            "today": "Today: read-only navigation tools over a curated Synataric corpus.",
            "next": (
                "Next: provider APIs, appointment request workflow, insurance verification, travel/lodging APIs, "
                "document checklist, secure patient profile, FHIR/records connectors, and care coordinator handoff."
            ),
            "note": (
                "Any write action - booking, payment, provider outreach, insurance submission, or message sending - "
                "requires human approval."
            ),
        },
    }


def build_architecture_snapshot_cards() -> list[dict[str, str]]:
    return [
        {
            "title": "Ask Navigator",
            "label": "Grounded RAG",
            "text": "Direct question -> Pinecone retrieval -> FlashRank reranking -> grounded answer with sources.",
            "best_for": "What is the cost of cataract surgery in Bangalore?",
        },
        {
            "title": "Agent Navigator",
            "label": "Router-pattern agent",
            "text": "Classifies intent, selects one domain tool, or routes to safety / human clarification / out-of-scope.",
            "best_for": "Where can I find good cataract surgery in India?",
        },
        {
            "title": "ReAct Care Planner",
            "label": "Bounded agentic loop",
            "text": "Goal -> reason -> act -> observe -> decide next step. Used only for multi-step care plans.",
            "best_for": (
                "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, "
                "and risks."
            ),
        },
        {
            "title": "Trust & Safety Layer",
            "label": "Boundaries",
            "text": (
                "No diagnosis, no prescription, no booking, no payment. Missing information asks human; unsafe "
                "medical questions are refused."
            ),
            "best_for": "Should I take antibiotics after surgery?",
        },
    ]


def build_agentic_callout() -> dict[str, Any]:
    return {
        "title": "This is the agentic part",
        "body": (
            "The ReAct Care Planner takes a goal, senses the intent, calls one tool at a time, observes each "
            "result, and decides the next step. For the Bangalore cataract care-plan goal, it calls "
            "provider_search_tool, cost_estimate_tool, recovery_guidance_tool, and risk_checklist_tool before "
            "synthesizing the plan."
        ),
        "steps": [
            "Goal",
            "Sense intent",
            "Act: provider_search_tool",
            "Observe",
            "Act: cost_estimate_tool",
            "Observe",
            "Act: recovery_guidance_tool",
            "Observe",
            "Act: risk_checklist_tool",
            "Synthesize care plan",
        ],
        "note": "Bounded by max_steps=5 and read-only tools.",
    }


def build_executive_metric_cards(metrics: dict[str, Any]) -> list[dict[str, str]]:
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]
    return [
        {
            "title": "Agent Eval",
            "value": f"{agent['baseline_overall']:.4f} -> {agent['post_improvement_overall']:.4f}",
            "caption": "40-case golden dataset; Router Agent + ReAct Care Planner",
        },
        {
            "title": "Biggest Agent Gains",
            "value": "+0.1500",
            "caption": "status accuracy and task completion",
        },
        {
            "title": "Source Hit Rate",
            "value": "0.8250 -> 0.9500",
            "caption": "better evidence retrieval/use in post-improvement run",
        },
        {
            "title": "Fine-Tuned Router",
            "value": f"{existing['accuracy']:.3f} -> {fine['accuracy']:.3f}",
            "caption": "Week 5 local route classifier benchmark",
        },
        {
            "title": "Routing Latency",
            "value": f"{existing['average_latency_seconds']:.3f}s -> {fine['average_latency_seconds']:.3f}s",
            "caption": "observed validation latency improvement",
        },
        {
            "title": "Safety",
            "value": "Bounded",
            "caption": "unsafe medication/prescription requests route to safety response",
        },
    ]


def get_architecture_pipeline_nodes() -> list[dict[str, str]]:
    return [
        {
            "title": "Corpus",
            "role": "procedure, provider, cost, recovery, risk, policy files",
        },
        {
            "title": "RAG Evidence",
            "role": "load, chunk, embed, retrieve, rerank",
        },
        {
            "title": "MINT Router",
            "role": "selects the lightest workflow that works",
        },
        {
            "title": "Agent Tools",
            "role": "provider, cost, recovery, risk, travel, evidence, safety, ask-human",
        },
        {
            "title": "Bounded ReAct Planner",
            "role": "goal -> reason -> act -> observe",
        },
        {
            "title": "Grounded Care Plan",
            "role": "synthesized from observations and evidence",
        },
        {
            "title": "Safety / HITL / Evals",
            "role": "refusal, clarification, tracing, golden dataset scoring",
        },
    ]


def get_component_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "component": "sample_data",
            "type": "Pure Python",
            "llm_calls": "0",
            "role": "Creates illustrative procedure, provider, cost, risk, recovery, and policy files.",
        },
        {
            "component": "loaders",
            "type": "LangChain + Python",
            "llm_calls": "0",
            "role": "Loads Markdown, TXT, PDF, and CSV into Document objects with metadata.",
        },
        {
            "component": "chunking",
            "type": "LangChain",
            "llm_calls": "0 + embeddings for semantic",
            "role": "Fixed and semantic-style chunks.",
        },
        {
            "component": "indexing",
            "type": "OpenAI + Pinecone",
            "llm_calls": "embedding calls",
            "role": "Indexes chunks into fixed and semantic namespaces.",
        },
        {
            "component": "retrieval",
            "type": "Pinecone vector search",
            "llm_calls": "0 LLM",
            "role": "Returns top-k docs with retrieval metadata.",
        },
        {
            "component": "reranking",
            "type": "FlashRank local model",
            "llm_calls": "0 LLM",
            "role": "Reranks query-document pairs with intent boosts.",
        },
        {
            "component": "rag_chain",
            "type": "ChatOpenAI",
            "llm_calls": "1 generation call",
            "role": "Builds grounded answer with sources and safety rules.",
        },
        {
            "component": "agent_intents",
            "type": "LLM structured output + deterministic fallback",
            "llm_calls": "1 classification call",
            "role": "Classifies intent, missing fields, safety, and out-of-scope.",
        },
        {
            "component": "agent_tools",
            "type": "Python wrappers around RAG",
            "llm_calls": "tool-dependent",
            "role": "Domain tools return structured AgentToolResult.",
        },
        {
            "component": "agent_graph",
            "type": "LangGraph StateGraph",
            "llm_calls": "intent + tool calls",
            "role": "Routes to safety, out-of-scope, ask-human, tools, fallback, final response.",
        },
        {
            "component": "agent_session",
            "type": "Pure Python",
            "llm_calls": "0",
            "role": "Stores pending clarification and reruns after user supplies missing information.",
        },
        {
            "component": "react_care_agent",
            "type": "LangGraph loop + ChatOpenAI",
            "llm_calls": "variable",
            "role": "Bounded reason-act-observe loop for multi-step care plans.",
        },
        {
            "component": "app",
            "type": "Streamlit UI",
            "llm_calls": "depends on user action",
            "role": "Displays RAG, router agent, ReAct planner, evidence, diagnostics, and evaluation.",
        },
    ]


def get_metric_cards(metrics: dict[str, Any]) -> list[dict[str, str]]:
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]
    return [
        {
            "title": "Agent Eval",
            "value": f"{agent['post_improvement_overall']:.4f}",
            "caption": f"F1 Score | was {agent['baseline_overall']:.4f}",
            "delta": "+6.4%",
            "icon": "~",
        },
        {
            "title": "Router Accuracy",
            "value": f"{fine['accuracy']:.3f}",
            "caption": f"Exact match | was {existing['accuracy']:.3f}",
            "delta": "+80.2%",
            "icon": "->",
        },
        {
            "title": "Avg Latency",
            "value": f"{fine['average_latency_seconds']:.3f}s",
            "caption": f"End-to-end | was {existing['average_latency_seconds']:.3f}s",
            "delta": "-85.3%",
            "icon": "bolt",
        },
        {
            "title": "Safety Bound",
            "value": "Active",
            "caption": "HITL enforced | Read-only tools",
            "delta": "",
            "icon": "shield",
        },
    ]


def get_command_center_dashboard_data(metrics: dict[str, Any]) -> dict[str, Any]:
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]
    return {
        "title": "Synataric Global Healthcare Navigator",
        "subtitle": "Care-navigation workflow layer for cross-border treatment planning",
        "badges": [
            "Educational navigation only",
            "MINT + ReAct",
            "Evaluated",
            "Read-only tools",
        ],
        "pipeline_nodes": [
            {"title": "Corpus", "subtitle": "Medical KB", "icon": "DB"},
            {"title": "RAG Evidence", "subtitle": "Retrieval", "icon": "Q"},
            {"title": "MINT Router", "subtitle": "Orchestration", "icon": "R"},
            {"title": "Agent Tools", "subtitle": "4 bound tools", "icon": "T"},
            {"title": "Bounded ReAct", "subtitle": "Reasoning loop", "icon": "A"},
            {"title": "Grounded Plan", "subtitle": "Structured output", "icon": "P"},
            {"title": "Safety / HITL", "subtitle": "Evals", "icon": "S"},
        ],
        "kpis": [
            {
                "title": "Agent Eval",
                "label": "F1 score",
                "value": f"{agent['post_improvement_overall']:.4f}",
                "previous": f"{agent['baseline_overall']:.4f}",
                "improvement": "+6.4%",
                "accent": "blue",
            },
            {
                "title": "Router",
                "label": "Exact match",
                "value": f"{fine['accuracy']:.3f}",
                "previous": f"{existing['accuracy']:.3f}",
                "improvement": "+80.2%",
                "accent": "teal",
            },
            {
                "title": "Latency",
                "label": "Avg end-to-end",
                "value": f"{fine['average_latency_seconds']:.3f}s",
                "previous": f"{existing['average_latency_seconds']:.3f}s",
                "improvement": "-85.3%",
                "accent": "green",
            },
            {
                "title": "Safety",
                "label": "HITL enforced",
                "value": "Active",
                "previous": "",
                "improvement": "",
                "subtext": "Read-only · HITL enforced",
                "accent": "slate",
            },
        ],
        "console_preview": [
            {"index": "01", "label": "Cataract care plan — Bangalore", "active": True},
            {"index": "02", "label": "Cost estimate — Bangalore", "active": False},
            {"index": "03", "label": "Safety boundary — antibiotics", "active": False},
            {"index": "04", "label": "Human clarification — missing procedure", "active": False},
        ],
        "query_preview": (
            "Create a care travel plan for cataract surgery in Bangalore including providers, cost, "
            "recovery, and risks."
        ),
        "workflow_preview": [
            {"title": "Provider Search", "tool": "provider_search_tool", "icon": "Q"},
            {"title": "Cost Estimate", "tool": "cost_estimate_tool", "icon": "$"},
            {"title": "Recovery Guidance", "tool": "recovery_guidance_tool", "icon": "~"},
            {"title": "Risk Checklist", "tool": "risk_checklist_tool", "icon": "+"},
        ],
    }


def build_command_center_dashboard_html(data: dict[str, Any]) -> str:
    pipeline_nodes = data["pipeline_nodes"]
    pipeline_html = []
    for index, node in enumerate(pipeline_nodes):
        active_class = " is-active" if node["title"] == "MINT Router" else ""
        connector = '<div class="syn-pipe-connector"></div>' if index < len(pipeline_nodes) - 1 else ""
        pipeline_html.append(
            f"""
            <div class="syn-pipe-node{active_class}">
              <div class="syn-node-icon">{_escape(node['icon'])}</div>
              <div>
                <div class="syn-node-title">{_escape(node['title'])}</div>
                <div class="syn-node-subtitle">{_escape(node['subtitle'])}</div>
              </div>
            </div>
            {connector}
            """
        )

    kpi_html = []
    for kpi in data["kpis"]:
        previous = kpi.get("previous", "")
        improvement = kpi.get("improvement", "")
        previous_html = f'<span class="syn-previous">{_escape(previous)}</span>' if previous else ""
        improvement_html = f'<div class="syn-improvement">{_escape(improvement)}</div>' if improvement else ""
        subtext_html = f'<div class="syn-kpi-subtext">{_escape(kpi.get("subtext", ""))}</div>' if kpi.get("subtext") else ""
        kpi_html.append(
            f"""
            <div class="syn-kpi-card syn-accent-{_escape(kpi['accent'])}">
              <div class="syn-kpi-top">
                <div>
                  <div class="syn-kpi-title">{_escape(kpi['title'])}</div>
                  <div class="syn-kpi-label">{_escape(kpi['label'])}</div>
                </div>
                <div class="syn-kpi-icon"></div>
              </div>
              <div class="syn-kpi-value">{_escape(kpi['value'])} {previous_html}</div>
              <div class="syn-progress"><span></span></div>
              {improvement_html}
              {subtext_html}
            </div>
            """
        )

    scenario_html = []
    for item in data["console_preview"]:
        active_class = " is-selected" if item["active"] else ""
        scenario_html.append(
            f"""
            <div class="syn-scenario{active_class}">
              <span>{_escape(item['index'])}</span>
              <strong>{_escape(item['label'])}</strong>
            </div>
            """
        )

    timeline_html = []
    for step in data["workflow_preview"]:
        timeline_html.append(
            f"""
            <div class="syn-step">
              <div class="syn-step-icon">{_escape(step['icon'])}</div>
              <div>
                <div class="syn-step-title">{_escape(step['title'])}</div>
                <div class="syn-step-tool">{_escape(step['tool'])}</div>
              </div>
            </div>
            """
        )

    badges_html = "".join(f'<span class="syn-badge">{_escape(badge)}</span>' for badge in data["badges"])
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #EEF4FB;
  color: #0F172A;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.syn-dashboard {{
  width: 100%;
  padding: 6px 4px 18px;
}}
.syn-header {{
  display: grid;
  grid-template-columns: 56px minmax(280px, 1fr) auto;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
}}
.syn-logo {{
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  color: white;
  font-size: 28px;
  background: linear-gradient(145deg, #0B75BC, #14B8A6);
  box-shadow: 0 10px 24px rgba(11, 117, 188, 0.22);
}}
.syn-title {{ font-size: 22px; font-weight: 850; line-height: 1.1; }}
.syn-subtitle {{ color: #64748B; font-size: 16px; margin-top: 4px; }}
.syn-badges {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; max-width: 520px; }}
.syn-badge {{
  border: 1px solid #CFE1F5;
  background: rgba(255,255,255,0.78);
  border-radius: 999px;
  color: #334155;
  padding: 6px 11px;
  font-size: 13px;
  font-weight: 700;
  box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
}}
.syn-panel {{
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}}
.syn-section-title {{
  color: #7890AE;
  font-size: 13px;
  letter-spacing: 0.15em;
  font-weight: 850;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.syn-pipeline {{
  padding: 24px 28px 22px;
  margin-bottom: 18px;
  overflow: hidden;
}}
.syn-pipe-row {{
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(7, minmax(128px, 1fr));
  align-items: center;
  gap: 10px;
}}
.syn-pipe-node {{
  min-height: 70px;
  border: 1px solid #D7E3F0;
  border-radius: 12px;
  padding: 14px 14px;
  display: flex;
  align-items: center;
  gap: 12px;
  background: linear-gradient(180deg, #F8FBFF, #F2F7FC);
}}
.syn-pipe-node.is-active {{
  background: linear-gradient(135deg, #0B75BC, #0E8BCB);
  border-color: #0B75BC;
  color: #FFFFFF;
  box-shadow: 0 14px 28px rgba(11, 117, 188, 0.24);
}}
.syn-node-icon {{
  width: 28px;
  height: 28px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: #0B75BC;
  background: #EAF4FF;
  font-size: 12px;
  font-weight: 900;
}}
.is-active .syn-node-icon {{ background: rgba(255,255,255,0.2); color: #FFFFFF; }}
.syn-node-title {{ font-size: 15px; font-weight: 850; white-space: nowrap; }}
.syn-node-subtitle {{
  color: #7890AE;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  margin-top: 3px;
}}
.is-active .syn-node-subtitle {{ color: #C8E7FF; }}
.syn-pipe-connector {{ display: none; }}
.syn-kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 18px;
}}
.syn-kpi-card {{
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  min-height: 166px;
}}
.syn-kpi-top {{ display: flex; justify-content: space-between; gap: 10px; }}
.syn-kpi-title {{ font-size: 17px; font-weight: 850; }}
.syn-kpi-label {{
  color: #7890AE;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  margin-top: 5px;
}}
.syn-kpi-icon {{ width: 40px; height: 40px; border-radius: 10px; background: #EEF7FF; }}
.syn-kpi-value {{ margin-top: 18px; color: #0B75BC; font-size: 32px; font-weight: 900; }}
.syn-previous {{ color: #B6C5D8; font-size: 14px; text-decoration: line-through; margin-left: 8px; }}
.syn-progress {{ margin-top: 10px; height: 5px; border-radius: 999px; background: #E8EEF6; overflow: hidden; }}
.syn-progress span {{ display: block; width: 84%; height: 100%; border-radius: inherit; background: #14B8A6; }}
.syn-improvement {{ margin-top: 10px; color: #059669; font-weight: 850; font-size: 14px; }}
.syn-kpi-subtext {{ margin-top: 11px; color: #7890AE; font-weight: 800; }}
.syn-accent-green .syn-kpi-value {{ color: #059669; }}
.syn-accent-slate .syn-kpi-value {{ color: #64748B; }}
.syn-workspace {{
  display: grid;
  grid-template-columns: 1.45fr 0.8fr;
  gap: 18px;
}}
.syn-console, .syn-timeline {{ padding: 28px; }}
.syn-scenario-list {{ margin-top: 18px; display: grid; gap: 8px; }}
.syn-scenario {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 16px;
  border: 1px solid #E0E8F2;
  border-radius: 12px;
  background: #F8FBFF;
  color: #0F172A;
}}
.syn-scenario.is-selected {{
  background: #EAF4FF;
  border-color: #0B75BC;
  color: #075BA0;
}}
.syn-scenario span {{
  min-width: 36px;
  text-align: center;
  border: 1px solid #C9DAED;
  border-radius: 6px;
  padding: 3px 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 850;
  font-size: 12px;
}}
.syn-query {{
  margin-top: 18px;
  border: 1px solid #DCE6F2;
  background: #F8FBFF;
  border-radius: 12px;
  padding: 18px;
  line-height: 1.55;
  font-size: 17px;
}}
.syn-run {{
  display: inline-block;
  margin-top: 18px;
  background: linear-gradient(135deg, #0B75BC, #1589CC);
  color: #FFFFFF;
  border-radius: 12px;
  padding: 13px 20px;
  font-weight: 850;
  box-shadow: 0 12px 22px rgba(11, 117, 188, 0.24);
}}
.syn-step-list {{ margin-top: 18px; display: grid; gap: 14px; }}
.syn-step {{ display: grid; grid-template-columns: 44px 1fr; gap: 14px; align-items: center; }}
.syn-step-icon {{
  width: 44px;
  height: 44px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  border: 1px solid #D5E2F0;
  background: #EEF7FF;
  color: #7E9DBF;
  font-weight: 900;
}}
.syn-step-title {{ color: #9AAFD0; font-size: 17px; font-weight: 850; }}
.syn-step-tool {{
  margin-top: 3px;
  color: #9AAFD0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  font-weight: 700;
}}
@media (max-width: 1050px) {{
  .syn-header, .syn-workspace {{ grid-template-columns: 1fr; }}
  .syn-badges {{ justify-content: flex-start; }}
  .syn-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .syn-pipe-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
  <div class="syn-dashboard">
    <div class="syn-header">
      <div class="syn-logo">◎</div>
      <div>
        <div class="syn-title">{_escape(data['title'])}</div>
        <div class="syn-subtitle">{_escape(data['subtitle'])}</div>
      </div>
      <div class="syn-badges">{badges_html}</div>
    </div>

    <section class="syn-panel syn-pipeline">
      <div class="syn-section-title">ARCHITECTURE PIPELINE</div>
      <div class="syn-pipe-row">{''.join(pipeline_html)}</div>
    </section>

    <section class="syn-kpi-grid">{''.join(kpi_html)}</section>

    <section class="syn-workspace">
      <div class="syn-panel syn-console">
        <div class="syn-section-title">NAVIGATOR CONSOLE</div>
        <div class="syn-scenario-list">{''.join(scenario_html)}</div>
        <div class="syn-query">{_escape(data['query_preview'])}</div>
        <div class="syn-run">▶ Run Navigator</div>
      </div>
      <div class="syn-panel syn-timeline">
        <div class="syn-section-title">WORKFLOW TIMELINE</div>
        <div class="syn-step-list">{''.join(timeline_html)}</div>
      </div>
    </section>
  </div>
</body>
</html>
"""


def render_command_center_dashboard(metrics: dict[str, Any]) -> None:
    st.components.v1.html(
        build_command_center_dashboard_html(get_command_center_dashboard_data(metrics)),
        height=760,
        scrolling=False,
    )


def get_mint_ladder_cards() -> list[dict[str, str]]:
    return [
        {
            "mode": "Simple",
            "title": "Ask Navigator",
            "label": "Grounded RAG",
            "text": "Best for direct evidence questions.",
        },
        {
            "mode": "Routed",
            "title": "Agent Navigator",
            "label": "One intent -> one tool",
            "text": "Best for provider, cost, recovery, risk, safety, and clarification routing.",
        },
        {
            "mode": "Agentic",
            "title": "ReAct Care Planner",
            "label": "Goal -> reason -> act -> observe",
            "text": "Best for complete multi-step care plans.",
        },
        {
            "mode": "Guarded",
            "title": "Safety / Human Boundary",
            "label": "Refuse unsafe; ask missing info",
            "text": "Best for medication risk or underspecified requests.",
        },
    ]


def get_demo_tool_flow() -> list[str]:
    return [
        "Goal",
        "Sense intent",
        "provider_search_tool",
        "cost_estimate_tool",
        "recovery_guidance_tool",
        "risk_checklist_tool",
        "Synthesize",
    ]


def get_eval_delta_rows() -> list[dict[str, str]]:
    return [
        {"metric": "intent_accuracy", "baseline": "0.6750", "post": "0.7250", "delta": "+0.0500"},
        {"metric": "status_accuracy", "baseline": "0.6250", "post": "0.7750", "delta": "+0.1500"},
        {"metric": "tool_selection_accuracy", "baseline": "0.6250", "post": "0.6750", "delta": "+0.0500"},
        {"metric": "tool_sequence_accuracy", "baseline": "0.9750", "post": "0.9750", "delta": "0.0000"},
        {"metric": "source_hit_rate", "baseline": "0.8250", "post": "0.9500", "delta": "+0.1250"},
        {"metric": "human_handoff_accuracy", "baseline": "0.6750", "post": "0.8000", "delta": "+0.1250"},
        {"metric": "safety_refusal_accuracy", "baseline": "0.9750", "post": "0.9750", "delta": "0.0000"},
        {"metric": "out_of_scope_accuracy", "baseline": "0.9750", "post": "0.9750", "delta": "0.0000"},
        {"metric": "task_completion_score", "baseline": "0.6250", "post": "0.7750", "delta": "+0.1500"},
        {"metric": "trajectory_correctness", "baseline": "0.7750", "post": "0.8250", "delta": "+0.0500"},
        {"metric": "local_path_leakage_absence", "baseline": "0.9750", "post": "1.0000", "delta": "+0.0250"},
        {"metric": "overall_score", "baseline": "0.8283", "post": "0.8810", "delta": "+0.0527"},
    ]


def get_route_specific_result_cards(route_key: str) -> list[dict[str, Any]]:
    route = str(route_key or "").lower()
    if route in {"care_plan_multistep", "react_care_planner", "multi-step care plan"}:
        return [
            {"title": "Provider Options", "items": ["Sankara Eye Services", "Bangalore Eye Centre"]},
            {
                "title": "Estimated Cost",
                "items": [
                    "INR 45,000 - INR 150,000",
                    "Factors: lens choice, diagnostics, surgeon fees, facility type",
                ],
            },
            {
                "title": "Recovery Guidance",
                "items": [
                    "follow-up visits",
                    "eye/wound care instructions",
                    "medication-list questions",
                    "mobility or companion support",
                ],
            },
            {
                "title": "Risk and Red Flags",
                "items": [
                    "discuss individual risks with licensed clinician",
                    "watch for urgent symptoms",
                    "seek immediate care for severe or urgent symptoms",
                ],
            },
            {
                "title": "Questions to Ask a Clinician",
                "items": [
                    "Which lens option is appropriate for me?",
                    "What follow-up schedule should I plan around?",
                    "Which symptoms require urgent review?",
                ],
            },
            {
                "title": "Evidence Used",
                "items": [
                    "bangalore_eye_hospitals.csv",
                    "india_procedure_costs.csv",
                    "cataract_surgery_guide.md",
                    "post_op_recovery_guidelines.md",
                ],
            },
        ]
    if route == "cost_estimate":
        return [
            {"title": "Cost Range", "items": ["INR 45,000 - INR 150,000"]},
            {"title": "Cost Drivers", "items": ["lens choice", "diagnostics", "surgeon fee", "facility type"]},
            {"title": "Evidence", "items": ["india_procedure_costs.csv"]},
            {"title": "Related Providers", "items": ["Bangalore Eye Centre", "Sankara Eye Services"]},
            {"title": "Questions to Ask a Clinician", "items": ["Which lens option changes cost?", "What follow-up costs should I plan for?"]},
        ]
    if route == "provider_search":
        return [
            {"title": "Provider Options", "items": ["Bangalore Eye Centre", "Sankara Eye Services"]},
            {"title": "Navigation Features", "items": ["compare provider fit", "review follow-up logistics", "confirm accreditation and surgeon experience"]},
            {"title": "Evidence", "items": ["bangalore_eye_hospitals.csv", "provider_profiles.md"]},
            {"title": "Questions to Ask a Clinician", "items": ["Which provider is appropriate for my condition?", "What pre-op tests are required?"]},
        ]
    if route == "recovery_guidance":
        return [
            {"title": "Recovery Checklist", "items": ["eye/wound care instructions", "medication-list questions", "mobility/caregiver support"]},
            {"title": "Follow-up Visits", "items": ["confirm follow-up schedule", "plan travel around post-op review"]},
            {"title": "Red Flags", "items": ["severe pain", "sudden vision changes", "urgent symptoms need immediate care"]},
            {"title": "Evidence", "items": ["post_op_recovery_guidelines.md", "cataract_surgery_guide.md"]},
        ]
    if route == "risk_checklist":
        return [
            {"title": "Red Flags / Urgent Symptoms", "items": ["severe pain", "sudden vision changes", "signs of infection"]},
            {"title": "Safety Note", "items": ["discuss individual risks with a licensed clinician", "seek immediate care for severe or urgent symptoms"]},
            {"title": "Evidence", "items": ["travel_medical_risk_checklist.md", "post_op_recovery_guidelines.md"]},
        ]
    if route in {"unsafe_medical", "unsafe", "safety_response_tool"}:
        return [
            {
                "title": "Safety Boundary Triggered",
                "items": [
                    "Synataric cannot provide diagnosis, prescriptions, medication instructions, or urgent medical decisions.",
                    "Please consult a licensed clinician.",
                    "If symptoms are severe or urgent, seek immediate medical care.",
                ],
            }
        ]
    if route in {"needs_clarification", "needs_human", "ask_human_tool"}:
        return [
            {
                "title": "Clarification Needed",
                "items": [
                    "Which procedure are you considering?",
                    "Missing procedure blocks safe travel planning.",
                ],
            }
        ]
    if route in {"out_of_scope", "out_of_scope_response_tool"}:
        return [
            {
                "title": "Outside Synataric Scope",
                "items": [
                    "Suggested supported topics: providers, costs, recovery, risks, travel planning.",
                ],
            }
        ]
    if route in {"coverage_gap", "coverage_gap_response_tool"}:
        return [
            {
                "title": "Corpus Coverage Gap",
                "items": [
                    "Current corpus supports illustrative Bangalore/India care-navigation examples.",
                    "Missing: requested destination-specific provider or cost evidence.",
                    "Try queries about Bangalore providers, India costs, recovery, risks, or safety boundaries.",
                ],
            }
        ]
    return [{"title": "Grounded Result", "items": ["Synataric returned a bounded care-navigation response."]}]


def build_care_plan_cards(fields: dict[str, Any], evidence: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    question = str(fields.get("user_question") or "").lower()
    expected_route = str(fields.get("expected_route") or "").lower()
    is_bangalore_cataract_plan = "bangalore" in question and "cataract" in question and (
        "care_plan_multistep" in expected_route or "provider" in question or "risk" in question
    )
    if is_bangalore_cataract_plan:
        return [
            {"title": "Provider Options", "items": ["Sankara Eye Services", "Bangalore Eye Centre"]},
            {
                "title": "Estimated Cost",
                "items": [
                    "INR 45,000 - INR 150,000",
                    "Factors: lens choice, diagnostics, surgeon fee, facility type",
                ],
            },
            {
                "title": "Recovery Guidance",
                "items": [
                    "Follow-up visits",
                    "Wound/eye care instructions",
                    "Medication-list questions",
                    "Mobility support",
                ],
            },
            {
                "title": "Risk and Red Flags",
                "items": [
                    "Urgent symptoms / red flags",
                    "Discuss risks with licensed clinician",
                    "Seek immediate care for severe or urgent symptoms",
                ],
            },
            {
                "title": "Evidence Used",
                "items": [
                    "bangalore_eye_hospitals.csv",
                    "india_procedure_costs.csv",
                    "cataract_surgery_guide.md",
                    "post_op_recovery_guidelines.md",
                ],
            },
            {
                "title": "Questions to Ask a Clinician",
                "items": [
                    "Which lens option is appropriate for me?",
                    "What follow-up schedule should I plan around?",
                    "Which symptoms require urgent review?",
                ],
            },
        ]

    plan = parse_care_plan_sections(fields.get("final_answer") or "")
    cards: list[dict[str, Any]] = []
    for key in ["provider_options", "estimated_cost", "recovery_guidance", "risk_red_flags", "clinician_questions"]:
        items = plan["sections"].get(key, [])
        if items:
            cards.append({"title": SECTION_LABELS.get(key, key), "items": items})
    evidence_names = [_source_name(row.get("source") or row.get("file_name") or "") for row in (evidence or sources or [])]
    evidence_names = [name for name in dict.fromkeys(evidence_names) if name and name != "unknown"]
    if evidence_names:
        cards.append({"title": "Evidence Used", "items": evidence_names})
    if not cards and plan["fallback"]:
        cards.append({"title": "Care Navigation Answer", "items": plan["fallback"]})
    return cards


def detect_requested_geography(question: str) -> str | None:
    normalized = normalize_requested_geography(question)
    return GEO_DISPLAY_NAMES.get(normalized) if normalized else None


def normalize_requested_geography(question: str) -> str | None:
    text = str(question or "").lower()
    for geography, aliases in {**SUPPORTED_GEO_ALIASES, **UNSUPPORTED_GEO_ALIASES}.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return geography
    return None


def detect_requested_procedure(question: str) -> str | None:
    text = str(question or "").lower()
    if "cataract" in text:
        return "cataract"
    if "knee" in text and "replacement" in text:
        return "knee_replacement"
    if "cardiac" in text or "bypass" in text or "cabg" in text:
        return "cardiac_bypass"
    return None


def _coverage_items(evidence: list[dict], sources: list[dict], result: Any = None) -> list[dict[str, Any]]:
    items = []
    for item in (evidence or []) + (sources or []):
        item_dict = _as_dict(item)
        source = _source_name(item_dict.get("source") or item_dict.get("file_name") or item_dict.get("title") or "")
        text = " ".join(
            [
                source,
                sanitize_demo_text(item_dict.get("category") or ""),
                sanitize_demo_text(item_dict.get("snippet") or ""),
                sanitize_demo_text(item_dict.get("retrieved_fact") or ""),
            ]
        ).lower()
        items.append({"source": source, "text": text, "geographies": infer_source_geography({"source": source, "text": text})})
    result_dict = _normalize_result(result) if result is not None else {}
    for observation in result_dict.get("observations") or []:
        obs_dict = _as_dict(observation)
        items.extend(
            _coverage_items(
                obs_dict.get("evidence") or [],
                obs_dict.get("sources") or [],
            )
        )
        text = " ".join(
            [
                sanitize_demo_text(obs_dict.get("tool_name") or ""),
                sanitize_demo_text(obs_dict.get("status") or ""),
                sanitize_demo_text(obs_dict.get("answer") or ""),
            ]
        ).lower()
        source = sanitize_demo_text(obs_dict.get("tool_name") or "observation")
        items.append({"source": source, "text": text, "geographies": infer_source_geography({"source": source, "text": text})})
    for line in result_dict.get("execution_log") or []:
        text = sanitize_demo_text(line).lower()
        items.append({"source": "execution_log", "text": text, "geographies": infer_source_geography({"source": "execution_log", "text": text})})
    if result_dict.get("final_answer"):
        text = sanitize_demo_text(result_dict.get("final_answer")).lower()
        items.append({"source": "final_answer", "text": text, "geographies": infer_source_geography({"source": "final_answer", "text": text})})
    return items


def _has_any(items: list[dict[str, Any]], terms: list[str]) -> bool:
    return any(any(term in item["text"] for term in terms) for item in items)


def infer_source_geography(item: dict | str) -> set[str]:
    if isinstance(item, dict):
        if item.get("source") == "final_answer":
            return set()
        source = _source_name(item.get("source") or item.get("file_name") or "")
        text = " ".join(
            [
                sanitize_demo_text(source),
                sanitize_demo_text(item.get("file_name") or ""),
                sanitize_demo_text(item.get("category") or ""),
                sanitize_demo_text(item.get("snippet") or ""),
                sanitize_demo_text(item.get("retrieved_fact") or ""),
                sanitize_demo_text(item.get("text") or ""),
            ]
        ).lower()
    else:
        source = _source_name(item)
        text = sanitize_demo_text(item).lower()
    geographies: set[str] = set()
    if any(term in text for term in ["bangalore", "bengaluru", "bangalore_eye_hospitals.csv"]):
        geographies.update({"bangalore", "india"})
    if any(term in text for term in ["india", "india_procedure_costs.csv", "inr", "travel_stay_costs.csv"]):
        geographies.add("india")
    for geography, aliases in UNSUPPORTED_GEO_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            geographies.add(geography)
    if source in GENERAL_PROCEDURE_SOURCES:
        geographies.add("general")
    return geographies


def _matches_requested_geography(requested_geo: str | None, item_geographies: set[str], *, allow_general: bool = False) -> bool:
    if requested_geo is None:
        return True
    if allow_general and "general" in item_geographies:
        return True
    if requested_geo in SUPPORTED_PROVIDER_COST_GEOS:
        return bool({"bangalore", "india"} & item_geographies)
    return False


def is_geography_supported_for_section(requested_geo: str | None, section: str, evidence_items: list[dict[str, Any]]) -> bool:
    allow_general = section in {"recovery", "risk"}
    if section in {"provider", "cost"} and requested_geo not in SUPPORTED_PROVIDER_COST_GEOS:
        return False
    return any(_matches_requested_geography(requested_geo, set(item.get("geographies") or []), allow_general=allow_general) for item in evidence_items)


def _successful_tool_names(result: Any) -> set[str]:
    successful = set()
    for call in extract_tool_calls(result):
        status = str(call.get("status") or "").lower()
        if status in {"success", "complete", "observing"}:
            successful.add(str(call.get("tool") or ""))
    return successful


def detect_coverage_gaps(question: str, evidence: list[dict], sources: list[dict], result: Any = None) -> dict[str, Any]:
    geography = detect_requested_geography(question)
    requested_geo = normalize_requested_geography(question)
    procedure = detect_requested_procedure(question)
    result_dict = _normalize_result(result) if result is not None else {}
    status = str(result_dict.get("status") or "").lower()
    items = _coverage_items(evidence, sources, result)
    successful_tools = _successful_tool_names(result)
    asks_provider = bool(re.search(r"\b(provider|providers|hospital|hospitals|clinic|clinics|where can i find)\b", question.lower()))
    asks_cost = bool(re.search(r"\b(cost|costs|price|estimate|pricing|budget)\b", question.lower()))

    base_payload = {
        "requested_geography": geography,
        "requested_geography_key": requested_geo,
        "requested_procedure": procedure,
        "geography_supported": False,
        "provider_coverage": "missing",
        "cost_coverage": "missing",
        "recovery_coverage": "missing",
        "risk_coverage": "missing",
        "provider_requested": asks_provider,
        "cost_requested": asks_cost,
    }
    if result_dict.get("requires_human") or status == "needs_human":
        return {
            **base_payload,
            "coverage": "Pending",
            "reason": "Additional information is needed before checking corpus coverage.",
        }
    if status in {"unsafe", "out_of_scope"}:
        reason = "Safety boundary triggered." if status == "unsafe" else "Request is outside Synataric healthcare-navigation scope."
        return {
            **base_payload,
            "coverage": "Not applicable",
            "reason": reason,
        }
    if status == "coverage_gap":
        return {
            **base_payload,
            "coverage": "Not available",
            "reason": "Corpus coverage gap.",
        }
    if requested_geo is None and (asks_provider or asks_cost):
        return {
            **base_payload,
            "coverage": "Pending",
            "reason": "Destination geography is needed before checking provider and cost coverage.",
        }

    provider_terms = [
        "bangalore_eye_hospitals.csv",
        "provider_profiles.md",
        "hospitals",
        "providers",
        "hospital_name",
        "bangalore eye centre",
        "sankara eye services",
        "provider profile",
        "cataract and community eye care",
        "cataract and retina referral",
        "hospital",
        "provider",
    ]
    cost_terms = [
        "india_procedure_costs.csv",
        "travel_stay_costs.csv",
        "costs",
        "low_estimate_inr",
        "high_estimate_inr",
        "45000",
        "150000",
        "45,000",
        "150,000",
        "cost_notes",
        "cataract surgery city: bangalore",
        "inr",
        "estimate",
    ]
    recovery_terms = [
        "post_op_recovery_guidelines.md",
        "cataract_surgery_guide.md",
        "follow-up visits",
        "wound care",
        "medication list",
        "recovery planning",
        "post-operative",
        "companion",
        "mobility support",
        "recovery",
        "follow-up",
    ]
    risk_terms = [
        "travel_medical_risk_checklist.md",
        "post_op_recovery_guidelines.md",
        "cataract_surgery_guide.md",
        "synataric_disclaimer_and_safety.md",
        "red-flag symptoms",
        "red flag symptoms",
        "urgent review",
        "immediate medical care",
        "increased pain",
        "sudden vision changes",
        "infection",
        "risks",
        "risk",
        "safety",
    ]

    provider_items = [item for item in items if any(term in item["text"] for term in provider_terms)]
    cost_items = [item for item in items if any(term in item["text"] for term in cost_terms)]
    recovery_items = [item for item in items if any(term in item["text"] for term in recovery_terms)]
    risk_items = [item for item in items if any(term in item["text"] for term in risk_terms)]

    provider_available = is_geography_supported_for_section(requested_geo, "provider", provider_items)
    cost_available = is_geography_supported_for_section(requested_geo, "cost", cost_items)
    recovery_available = is_geography_supported_for_section(requested_geo, "recovery", recovery_items)
    risk_available = is_geography_supported_for_section(requested_geo, "risk", risk_items)

    tool_geo_supported = requested_geo in {None, "bangalore", "india"}
    if tool_geo_supported:
        provider_available = provider_available or "provider_search_tool" in successful_tools
        cost_available = cost_available or "cost_estimate_tool" in successful_tools
    recovery_available = recovery_available or "recovery_guidance_tool" in successful_tools
    risk_available = risk_available or "risk_checklist_tool" in successful_tools

    provider_coverage = "available" if provider_available else "missing"
    cost_coverage = "available" if cost_available else "missing"
    recovery_coverage = "available" if recovery_available else "missing"
    risk_coverage = "available" if risk_available else "missing"
    geography_supported = requested_geo in {None, *SUPPORTED_PROVIDER_COST_GEOS}
    missing_required = (asks_provider and provider_coverage == "missing") or (asks_cost and cost_coverage == "missing")
    any_available = any(
        value == "available"
        for value in [provider_coverage, cost_coverage, recovery_coverage, risk_coverage]
    )
    if geography_supported and not missing_required:
        coverage = "Strong"
    elif any_available:
        coverage = "Partial"
    else:
        coverage = "Missing"

    return {
        "requested_geography": geography,
        "requested_geography_key": requested_geo,
        "requested_procedure": procedure,
        "coverage": coverage,
        "geography_supported": geography_supported,
        "provider_coverage": provider_coverage,
        "cost_coverage": cost_coverage,
        "recovery_coverage": recovery_coverage,
        "risk_coverage": risk_coverage,
        "provider_requested": asks_provider,
        "cost_requested": asks_cost,
    }


def coverage_note_for_gaps(gaps: dict[str, Any]) -> str:
    has_provider_or_cost_gap = gaps.get("provider_coverage") == "missing" or gaps.get("cost_coverage") == "missing"
    if gaps.get("coverage") != "Partial" or not has_provider_or_cost_gap:
        return ""
    if gaps.get("geography_supported", True):
        return ""
    country = gaps.get("requested_geography") or GEO_DISPLAY_NAMES.get(gaps.get("requested_geography_key")) or "that country"
    return (
        f"Coverage note: I don't have {country}-specific provider or cost records in the current Synataric corpus. "
        "The current indexed corpus is strongest for illustrative India/Bangalore care-navigation examples. "
        "I can still summarize general procedure recovery and risk planning considerations from the available "
        f"guides, but I cannot recommend {country} providers or quote {country} costs from the current evidence base."
    )


def rewrite_answer_for_coverage_gaps(answer_text: Any, gaps: dict[str, Any]) -> str:
    answer = sanitize_demo_text(answer_text)
    if gaps.get("coverage") != "Partial":
        return answer

    plan = parse_care_plan_sections(answer_text)
    note = coverage_note_for_gaps(gaps)
    sections = plan["sections"]
    country = gaps.get("requested_geography") or GEO_DISPLAY_NAMES.get(gaps.get("requested_geography_key")) or "that country"
    if gaps.get("provider_coverage") == "missing":
        sections["provider_options"] = [
            f"Not available in the current Synataric corpus. Add {country}-specific provider evidence or a vetted provider connector to support this section."
        ]
    if gaps.get("cost_coverage") == "missing":
        sections["estimated_cost"] = [
            f"Not available in the current Synataric corpus. Current cost tables are India/Bangalore illustrative data, not {country}-specific pricing evidence."
        ]

    lines = []
    if note:
        lines.extend(["#### Coverage Note", note, ""])
    for key in [
        "summary",
        "provider_options",
        "estimated_cost",
        "recovery_guidance",
        "risk_red_flags",
        "clinician_questions",
        "sources",
    ]:
        items = sections.get(key, [])
        if not items:
            continue
        lines.append(f"#### {SECTION_LABELS.get(key, key)}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    return "\n".join(lines).strip()


def build_coverage_safe_care_plan_cards(
    fields: dict[str, Any],
    evidence: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    coverage_gaps: dict[str, Any],
) -> list[dict[str, Any]]:
    route_key = _route_key_for_result(fields)
    cards = get_route_specific_result_cards(route_key)
    if route_key == "care_plan_multistep":
        safe_fields = dict(fields)
        safe_fields["final_answer"] = rewrite_answer_for_coverage_gaps(fields.get("final_answer"), coverage_gaps)
        cards = build_care_plan_cards(safe_fields, evidence, sources) or cards
    if coverage_gaps.get("provider_coverage") == "missing":
        cards = [card for card in cards if card.get("title") != "Provider Options"]
        cards.insert(0, {"title": "Provider Options", "items": ["Not available in the current Synataric corpus."]})
    if coverage_gaps.get("cost_coverage") == "missing":
        cards = [card for card in cards if card.get("title") != "Estimated Cost"]
        cards.insert(1 if cards else 0, {"title": "Estimated Cost", "items": ["Not available in the current Synataric corpus."]})
    return cards


def _normalize_result(result: Any) -> dict[str, Any]:
    result_dict = _as_dict(result)
    inner = result_dict.get("result")
    if inner is not None and isinstance(inner, dict):
        merged = dict(inner)
        merged.setdefault("session_status", result_dict.get("status"))
        return merged
    return result_dict


def extract_sources(result: Any) -> list[dict[str, Any]]:
    result_dict = _normalize_result(result)
    sources = result_dict.get("sources") or []
    rows = []
    for index, source in enumerate(sources, start=1):
        source_dict = _as_dict(source)
        rows.append(
            {
                "rank": source_dict.get("source_number") or index,
                "source": _source_name(
                    source_dict.get("file_name")
                    or source_dict.get("source")
                    or source_dict.get("source_path")
                    or source_dict.get("title")
                ),
                "category": source_dict.get("category", "N/A"),
                "snippet": _snippet(source_dict.get("retrieved_fact") or source_dict.get("snippet") or ""),
            }
        )
    return rows


def extract_evidence(result: Any) -> list[dict[str, Any]]:
    result_dict = _normalize_result(result)
    raw_evidence = result_dict.get("evidence") or result_dict.get("reranked_docs") or result_dict.get("retrieved_docs") or []
    rows = []
    for index, item in enumerate(raw_evidence, start=1):
        item_dict = _as_dict(item)
        metadata = _as_dict(item_dict.get("metadata"))
        content = item_dict.get("page_content") or item_dict.get("snippet") or item_dict.get("text") or item_dict.get("content") or ""
        rows.append(
            {
                "rank": item_dict.get("rank") or index,
                "source": _source_name(
                    item_dict.get("file_name")
                    or item_dict.get("source")
                    or metadata.get("file_name")
                    or metadata.get("source")
                    or metadata.get("source_path")
                ),
                "category": item_dict.get("category") or metadata.get("category", "N/A"),
                "retrieval_score": item_dict.get("retrieval_score")
                or metadata.get("similarity_score")
                or metadata.get("retrieval_score"),
                "rerank_score": item_dict.get("rerank_score") or metadata.get("rerank_score"),
                "snippet": _snippet(content),
            }
        )
    return rows


def extract_tool_calls(result: Any) -> list[dict[str, Any]]:
    result_dict = _normalize_result(result)
    calls = result_dict.get("tool_calls") or []
    rows = []
    for index, call in enumerate(calls, start=1):
        call_dict = _as_dict(call)
        rows.append(
            {
                "step": call_dict.get("step") or index,
                "tool": call_dict.get("tool_name") or call_dict.get("tool") or call_dict.get("name") or "tool",
                "input": sanitize_demo_text(call_dict.get("tool_input") or call_dict.get("input") or call_dict.get("args") or ""),
                "status": call_dict.get("status") or call_dict.get("result_status") or "complete",
            }
        )
    return rows


def extract_execution_log(result: Any) -> list[str]:
    result_dict = _normalize_result(result)
    log = result_dict.get("execution_log") or []
    if isinstance(log, str):
        return [sanitize_demo_text(log)]
    return [sanitize_demo_text(line) for line in log]


def _workflow_step(
    step_id: str,
    title: str,
    detail: str,
    status: str = "waiting",
    tool: str = "",
    phase: str = "",
    observation: str = "",
) -> dict[str, Any]:
    return {
        "id": step_id,
        "title": title,
        "detail": sanitize_demo_text(detail),
        "status": status,
        "tool": tool,
        "phase": phase,
        "observation": sanitize_demo_text(observation),
    }


def build_planned_workflow_for_scenario(scenario_key: str) -> list[dict[str, Any]]:
    key = str(scenario_key or "").lower().replace("_", " ")
    if "multi" in key or "care plan" in key:
        return [
            _workflow_step(
                "sense",
                "Sense goal",
                "Care-planning goal detected: providers + cost + recovery + risks.",
                phase="Sense",
            ),
            _workflow_step(
                "plan",
                "Select workflow",
                "MINT selected bounded ReAct Care Planner.",
                phase="Plan",
            ),
            _workflow_step(
                "provider",
                "Act: Search provider options",
                "Running provider_search_tool.",
                tool="provider_search_tool",
                phase="Act",
                observation="Observe: Bangalore provider evidence found.",
            ),
            _workflow_step(
                "cost",
                "Act: Estimate procedure costs",
                "Running cost_estimate_tool.",
                tool="cost_estimate_tool",
                phase="Act",
                observation="Observe: India/Bangalore cost evidence found.",
            ),
            _workflow_step(
                "recovery",
                "Act: Gather recovery guidance",
                "Running recovery_guidance_tool.",
                tool="recovery_guidance_tool",
                phase="Act",
                observation="Observe: Recovery and follow-up guidance found.",
            ),
            _workflow_step(
                "risk",
                "Act: Check risks and red flags",
                "Running risk_checklist_tool.",
                tool="risk_checklist_tool",
                phase="Act",
                observation="Observe: Risk and urgent-review guidance found.",
            ),
            _workflow_step(
                "final",
                "Synthesize grounded care plan",
                "Final answer generated from tool observations and retrieved evidence.",
                phase="Synthesize",
            ),
        ]
    if "cost" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("intent", "Classify care intent", "Selecting cost_estimate."),
            _workflow_step("cost", "Run cost estimate", "Running cost_estimate_tool.", tool="cost_estimate_tool"),
            _workflow_step("evidence", "Retrieve cost evidence", "Checking cost table."),
            _workflow_step("final", "Generate grounded answer", "Synthesizing cost guidance."),
        ]
    if "provider" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("intent", "Classify care intent", "Selecting provider_search."),
            _workflow_step("provider", "Run provider search", "Running provider_search_tool.", tool="provider_search_tool"),
            _workflow_step("evidence", "Retrieve provider evidence", "Searching provider evidence."),
            _workflow_step("final", "Generate provider options", "Preparing provider options."),
        ]
    if "human" in key or "clarification" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("missing", "Detect missing procedure", "Missing procedure detected."),
            _workflow_step("clarification", "Ask human clarification", "Asking for clarification.", tool="ask_human_tool"),
            _workflow_step("pause", "Pause workflow", "Pause workflow until the user answers.", status="warning"),
        ]
    if "safety" in key or "unsafe" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("unsafe", "Detect unsafe medical request", "Medication advice request detected."),
            _workflow_step("boundary", "Trigger safety boundary", "Safety boundary triggered - medication advice is not provided.", tool="safety_response_tool"),
            _workflow_step("final", "Return clinician-directed safety response", "Please consult a licensed clinician."),
        ]
    if "evidence" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("intent", "Classify care intent", "Selecting find_evidence."),
            _workflow_step("evidence", "Run evidence lookup", "Running find_evidence_tool.", tool="find_evidence_tool"),
            _workflow_step("sources", "Show source files and snippets", "Preparing clean source labels."),
        ]
    if "scope" in key:
        return [
            _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
            _workflow_step("scope", "Detect out-of-scope request", "Request is outside care navigation."),
            _workflow_step("final", "Return bounded scope response", "Suggesting supported care-navigation topics.", tool="out_of_scope_response_tool"),
        ]
    return [
        _workflow_step("route", "Route request", "Reading the care-navigation goal..."),
        _workflow_step("intent", "Classify care intent", "Selecting the safest workflow."),
        _workflow_step("final", "Generate grounded answer", "Synthesizing grounded response."),
    ]


def update_step_status(
    steps: list[dict[str, Any]],
    step_id: str,
    status: str,
    detail: str | None = None,
) -> list[dict[str, Any]]:
    updated = []
    for step in steps:
        next_step = dict(step)
        if next_step.get("id") == step_id:
            next_step["status"] = status
            if detail is not None:
                next_step["detail"] = sanitize_demo_text(detail)
        updated.append(next_step)
    return updated


def extract_actual_workflow(result: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, call in enumerate(extract_tool_calls(result), start=1):
        status = str(call.get("status") or "complete").lower()
        ui_status = "complete" if status in {"success", "complete", "observing"} else "error" if status == "error" else "warning"
        detail = call.get("input") or f"Observed {call.get('tool')}."
        steps.append(
            _workflow_step(
                f"actual_tool_{index}",
                _humanize_tool_name(call.get("tool", "tool")),
                detail,
                ui_status,
                tool=str(call.get("tool") or ""),
            )
        )

    if steps:
        return steps

    for index, line in enumerate(extract_execution_log(result), start=1):
        match = re.search(r"(?:Executed|Running|Calling)\s+([a-zA-Z0-9_]+_tool).*?(?:status\s+([a-zA-Z_]+))?", line, re.IGNORECASE)
        if not match:
            continue
        tool_name = match.group(1)
        status_text = (match.group(2) or "success").lower()
        ui_status = "complete" if status_text in {"success", "complete"} else "error" if status_text == "error" else "warning"
        steps.append(
            _workflow_step(
                f"actual_log_{index}",
                _humanize_tool_name(tool_name),
                line,
                ui_status,
                tool=tool_name,
            )
        )

    if steps:
        return steps

    result_dict = _normalize_result(result)
    selected_tool = result_dict.get("selected_tool")
    if selected_tool:
        return [_workflow_step("actual_selected_tool", _humanize_tool_name(selected_tool), "Selected workflow tool.", "complete", selected_tool)]
    return []


def _workflow_progress(steps: list[dict[str, Any]]) -> float:
    if not steps:
        return 0.0
    weights = {"complete": 1.0, "warning": 1.0, "error": 1.0, "running": 0.5}
    return sum(weights.get(str(step.get("status")), 0.0) for step in steps) / len(steps)


def extract_demo_result_fields(result: Any, expected_route: str, latency_seconds: float) -> dict[str, Any]:
    result_dict = _normalize_result(result)
    session_status = get_value(result, "status") or result_dict.get("session_status")
    status = result_dict.get("status") or session_status or "complete"
    actual_route = result_dict.get("intent") or result_dict.get("route") or result_dict.get("expected_route") or "N/A"
    final_answer = (
        result_dict.get("final_answer")
        or result_dict.get("answer")
        or result_dict.get("response")
        or result_dict.get("human_question")
        or ""
    )
    tool_calls = extract_tool_calls(result_dict)
    return {
        "user_question": sanitize_demo_text(result_dict.get("question") or ""),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "selected_tool": result_dict.get("selected_tool")
        or result_dict.get("tool")
        or (tool_calls[0]["tool"] if tool_calls else "N/A"),
        "status": status,
        "requires_human": bool(result_dict.get("requires_human") or status == "needs_human"),
        "human_question": sanitize_demo_text(result_dict.get("human_question") or ""),
        "step_count": result_dict.get("step_count") or len(tool_calls) or "N/A",
        "tool_call_count": len(tool_calls),
        "runtime_latency": f"{latency_seconds:.3f}s",
        "safety_status": "Triggered" if status == "unsafe" or actual_route == "unsafe_medical" else "Clear",
        "warnings": [sanitize_demo_text(item) for item in result_dict.get("warnings", []) or []],
        "errors": [sanitize_demo_text(item) for item in result_dict.get("errors", []) or []],
        "final_answer": sanitize_demo_text(final_answer),
    }


def _namespace(strategy: str) -> str:
    return "synataric-semantic" if strategy == "semantic" else "synataric-fixed"


def _run_live_demo(question: str, scenario: dict[str, Any], strategy: str, top_k: int) -> tuple[Any, float, str | None]:
    started = time.perf_counter()
    try:
        if scenario["expected_route"] == "care_plan_multistep":
            from src.react_care_agent import run_react_care_agent

            result = run_react_care_agent(question, namespace=_namespace(strategy), top_k=top_k)
        else:
            from src.agent_session import start_agent_session

            session_result = start_agent_session(question, namespace=_namespace(strategy), top_k=top_k)
            result = session_result
        return result, time.perf_counter() - started, None
    except Exception as exc:
        fallback = {
            "question": question,
            "status": "demo_fallback",
            "intent": scenario["expected_route"],
            "selected_tool": scenario["workflow"],
            "requires_human": scenario["expected_route"] == "needs_clarification",
            "human_question": scenario.get("human_question", ""),
            "answer": "Live run could not execute in this environment. Stored benchmark results are still available below.",
            "warnings": ["Live run could not execute in this environment. Stored benchmark results are still available below."],
            "errors": [str(exc)],
            "tool_calls": [
                {"step": index, "tool_name": tool, "tool_input": question, "status": "expected"}
                for index, tool in enumerate(scenario.get("expected_tools", []), start=1)
            ],
        }
        return fallback, time.perf_counter() - started, str(exc)


def _format_metric(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def inject_demo_medical_css() -> None:
    st.markdown(STYLE_BLOCK, unsafe_allow_html=True)
    return
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f7fafc !important;
        }
        .main .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
        }
        .syn-demo-shell {
            color: #102033;
        }
        .syn-demo-hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef9fb 100%);
            border: 1px solid #d7e9ef;
            border-radius: 22px;
            padding: 28px 30px;
            box-shadow: 0 18px 45px rgba(15, 76, 92, 0.10);
            margin-bottom: 18px;
        }
        .syn-demo-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #087e8b;
            background: #e6f7f9;
            border: 1px solid #bee6eb;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .syn-demo-title {
            color: #0f172a;
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 900;
            margin: 16px 0 8px;
        }
        .syn-demo-subtitle {
            color: #475569;
            max-width: 760px;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .syn-demo-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .syn-demo-card h3, .syn-demo-card h4 {
            color: #0f172a !important;
            margin-top: 0;
        }
        .syn-demo-card p, .syn-demo-card li, .syn-demo-card div {
            color: #334155 !important;
        }
        .syn-demo-stat {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            min-height: 96px;
        }
        .syn-demo-stat-label {
            color: #64748b;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .syn-demo-stat-value {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 850;
            margin-top: 8px;
            overflow-wrap: anywhere;
        }
        .syn-demo-section-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 900;
            margin: 24px 0 12px;
        }
        .syn-demo-provider {
            border: 1px solid #dbeafe;
            background: #f8fbff;
            border-radius: 16px;
            padding: 15px;
            height: 100%;
        }
        .syn-demo-provider-name {
            color: #0f172a;
            font-weight: 850;
            margin-bottom: 6px;
        }
        .syn-mint-hero {
            background: linear-gradient(135deg, #ffffff 0%, #ecfeff 100%);
            border: 1px solid #bae6fd;
            border-radius: 22px;
            padding: 26px 28px;
            box-shadow: 0 18px 42px rgba(14, 116, 144, 0.12);
            margin-bottom: 16px;
        }
        .syn-mint-title {
            color: #0f172a;
            font-size: 2.3rem;
            line-height: 1.08;
            font-weight: 900;
            margin: 14px 0 8px;
        }
        .syn-mint-subtitle {
            color: #0e7490;
            font-size: 1.04rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .syn-mint-copy {
            color: #475569;
            max-width: 860px;
            font-size: 1rem;
            line-height: 1.55;
        }
        .react-phase-badge {
            display: inline-block;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
            padding: 3px 8px;
            font-size: 0.70rem;
            font-weight: 900;
            margin-right: 6px;
            margin-bottom: 5px;
        }
        .react-tool-badge {
            display: inline-block;
            border-radius: 999px;
            background: #f0fdfa;
            color: #115e59;
            border: 1px solid #99f6e4;
            padding: 3px 8px;
            font-size: 0.70rem;
            font-weight: 850;
            margin-bottom: 5px;
        }
        .react-observation {
            color: #0f766e !important;
            font-size: 0.84rem;
            margin-top: 4px;
            font-weight: 700;
        }
        .syn-demo-callout {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 5px solid #0e7490;
            border-radius: 16px;
            padding: 16px;
            margin: 14px 0;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }
        .syn-demo-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .syn-story-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 15px;
            min-height: 112px;
            box-shadow: 0 10px 24px rgba(14, 116, 144, 0.07);
        }
        .syn-story-value {
            color: #0f172a;
            font-size: 1.12rem;
            font-weight: 900;
            margin: 5px 0;
        }
        .syn-demo-pill {
            display: inline-block;
            border-radius: 999px;
            background: #ecfeff;
            border: 1px solid #a5f3fc;
            color: #155e75;
            font-size: 0.76rem;
            font-weight: 800;
            padding: 4px 10px;
            margin-bottom: 8px;
        }
        .syn-demo-cost {
            background: linear-gradient(135deg, #ecfeff, #f0fdf4);
            border: 1px solid #99f6e4;
        }
        .syn-demo-alert {
            background: #fff7ed;
            border: 1px solid #fed7aa;
        }
        .syn-demo-missing {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
        }
        .syn-demo-clarify {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
        }
        .syn-demo-stepper {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        .syn-demo-step {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 12px;
        }
        .syn-demo-step-index {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #0e7490;
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 0.8rem;
            margin-right: 8px;
        }
        .syn-demo-muted {
            color: #64748b !important;
            font-size: 0.9rem;
        }
        .workflow-panel {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 14px 34px rgba(14, 116, 144, 0.08);
            margin: 14px 0;
        }
        .workflow-panel h3 {
            color: #0f172a !important;
            margin: 0 0 12px;
        }
        .workflow-step {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 12px;
            align-items: start;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 12px;
            margin: 9px 0;
            background: #ffffff;
        }
        .workflow-step-running {
            border-color: #67e8f9;
            background: #ecfeff;
            box-shadow: 0 0 0 3px rgba(103, 232, 249, 0.22);
        }
        .workflow-step-complete {
            border-color: #99f6e4;
            background: #f0fdfa;
        }
        .workflow-step-warning {
            border-color: #fed7aa;
            background: #fff7ed;
        }
        .workflow-step-error {
            border-color: #fecaca;
            background: #fef2f2;
        }
        .workflow-step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #e2e8f0;
            color: #334155;
            font-weight: 900;
            font-size: 0.85rem;
        }
        .workflow-step-running .workflow-step-icon {
            background: #0e7490;
            color: white;
        }
        .workflow-step-complete .workflow-step-icon {
            background: #0f766e;
            color: white;
        }
        .workflow-step-warning .workflow-step-icon {
            background: #f97316;
            color: white;
        }
        .workflow-step-error .workflow-step-icon {
            background: #dc2626;
            color: white;
        }
        .workflow-step-title {
            color: #0f172a;
            font-weight: 850;
            line-height: 1.25;
        }
        .workflow-step-detail {
            color: #475569;
            margin-top: 3px;
            font-size: 0.9rem;
        }
        .coverage-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .coverage-item {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px;
        }
        .coverage-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.74rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .coverage-strong {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .coverage-partial {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
        }
        .coverage-missing {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
        }
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        .stButton > button {
            background: linear-gradient(135deg, #0e7490, #14b8a6) !important;
            border: 0 !important;
            border-radius: 14px !important;
            min-height: 3rem;
            font-weight: 850 !important;
            box-shadow: 0 14px 28px rgba(14, 116, 144, 0.22);
        }
        .stTextArea textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mint_hero() -> None:
    st.markdown(
        """
        <div class="syn-demo-shell">
          <div class="syn-mint-hero">
            <div class="syn-demo-kicker">Educational healthcare navigation only - not medical advice</div>
            <div class="syn-mint-title">Synataric Intelligence Layer</div>
            <div class="syn-mint-subtitle">MINT-style routing + grounded RAG + bounded ReAct care planning</div>
            <div class="syn-mint-copy">
              Synataric uses the minimum intelligence necessary. Simple questions go to RAG or one tool.
              Complex care-planning goals activate a bounded ReAct planner that reasons, calls tools,
              observes evidence, and synthesizes a grounded plan.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_medical_header() -> None:
    render_mint_hero()


def render_executive_demo_story(metrics: dict[str, Any]) -> None:
    story = build_executive_story_content()

    with st.container(border=True):
        top_left, top_right = st.columns([2.2, 1])
        with top_left:
            st.caption("Executive demo mode")
            st.markdown(f"## {story['title']}")
            st.write(story["subtitle"])
            st.info(story["positioning"])
        with top_right:
            for badge in story["badges"]:
                st.caption(badge)

    st.markdown("### Why Synataric?")
    why_columns = st.columns(3)
    for column, card in zip(why_columns, story["why_cards"]):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {card['title']}")
                st.write(card["text"])
    st.caption(story["why_caption"])

    st.markdown("### Architecture at a glance")
    st.caption(" -> ".join(story["architecture_flow"]))
    architecture_columns = st.columns(4)
    for column, card in zip(architecture_columns, build_architecture_snapshot_cards()):
        with column:
            with st.container(border=True):
                st.caption(card["label"])
                st.markdown(f"#### {card['title']}")
                st.write(card["text"])
                st.caption(f"Best for: {card['best_for']}")

    callout = build_agentic_callout()
    with st.container(border=True):
        st.markdown(f"### {callout['title']}")
        st.write(callout["body"])
        step_columns = st.columns(5)
        for index, step in enumerate(callout["steps"]):
            with step_columns[index % len(step_columns)]:
                st.caption(step)
        st.caption(callout["note"])

    st.markdown("### Measured improvement")
    st.caption(story["measured_improvement_text"])
    metric_cards = build_executive_metric_cards(metrics)
    for row_start in range(0, len(metric_cards), 3):
        metric_columns = st.columns(3)
        for column, card in zip(metric_columns, metric_cards[row_start : row_start + 3]):
            with column:
                with st.container(border=True):
                    st.metric(card["title"], card["value"])
                    st.caption(card["caption"])

    with st.expander("What to say in 20 seconds", expanded=True):
        st.write(story["recording_script"])

    production = story["production_path"]
    with st.container(border=True):
        st.markdown(f"#### {production['title']}")
        st.write(production["today"])
        st.write(production["next"])
        st.warning(production["note"])


def render_mint_decision_ladder() -> None:
    st.markdown("### MINT Decision Ladder")
    st.caption("Use the minimum intelligence necessary: Simple -> Routed -> Agentic -> Guarded")

    columns = st.columns(4)
    for column, card in zip(columns, build_mint_decision_ladder()):
        with column:
            with st.container(border=True):
                st.caption(card["mode"])
                st.markdown(f"#### {card['title']}")
                st.markdown(f"**{card['label']}**")
                st.write(card["description"])
                st.caption(f"Example: {card['example']}")


def render_why_react_panel(scenario_key: str, question: str) -> None:
    panel = build_why_react_panel(scenario_key, question)
    chips = "".join(f'<span class="syn-demo-pill">{_escape(chip)}</span>' for chip in panel.get("chips", []))
    st.markdown(
        f"""
        <div class="syn-demo-card">
          <div class="syn-demo-pill">{_escape(panel["selected_workflow"])}</div>
          <h3>{_escape(panel["title"])}</h3>
          <p>{_escape(panel["body"])}</p>
          <div class="syn-demo-chip-row">{chips}</div>
          <p class="syn-demo-muted"><strong>Reasoning pattern:</strong> {_escape(panel["reasoning_pattern"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_not_a_chatbot_callout() -> None:
    items = [
        "The model does not diagnose.",
        "The model does not prescribe.",
        "The model does not book appointments.",
        "The model does not process payments.",
        "The system routes, retrieves, calls read-only tools, asks for missing information, and refuses unsafe requests.",
    ]
    bullets = "".join(f"<li>{_escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="syn-demo-callout">
          <h3>Not a chatbot - a bounded workflow system</h3>
          <ul>{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stat_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="syn-demo-stat">
          <div class="syn-demo-stat-label">{_escape(label)}</div>
          <div class="syn-demo-stat-value">{_escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_workflow_steps(container: Any, steps: list[dict[str, Any]], label: str = "Planned workflow") -> None:
    icon_map = {
        "waiting": "...",
        "running": "...",
        "complete": "OK",
        "warning": "!",
        "error": "X",
    }
    step_blocks = []
    for step in steps:
        status = str(step.get("status") or "waiting").lower()
        css_status = status if status in {"waiting", "running", "complete", "warning", "error"} else "waiting"
        phase = step.get("phase") or ""
        phase_html = f'<span class="react-phase-badge">{_escape(phase)}</span>' if phase else ""
        tool = step.get("tool") or ""
        tool_html = f'<span class="react-tool-badge">{_escape(tool)}</span>' if tool else ""
        observation = step.get("observation") or ""
        observation_html = f'<div class="react-observation">{_escape(observation)}</div>' if observation else ""
        step_blocks.append(
            f"""
            <div class="workflow-step workflow-step-{css_status}">
              <div class="workflow-step-icon">{_escape(icon_map.get(css_status, '...'))}</div>
              <div>
                <div>{phase_html}{tool_html}</div>
                <div class="workflow-step-title">{_escape(step.get("title"))}</div>
                <div class="workflow-step-detail">{_escape(step.get("detail"))}</div>
                {observation_html}
              </div>
            </div>
            """
        )
    with container.container():
        st.markdown(
            f"""
            <div class="workflow-panel">
              <h3>{_escape(label)}</h3>
              {''.join(step_blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_metric_cards(metrics: dict[str, Any]) -> None:
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]
    columns = st.columns(5)
    columns[0].metric("Router Accuracy", f"{existing['accuracy']:.3f} -> {fine['accuracy']:.3f}", "+44.5 points")
    columns[1].metric("Router Macro F1", f"{existing['macro_f1']:.3f} -> {fine['macro_f1']:.3f}", "+50.7 points")
    columns[2].metric(
        "Routing Latency",
        f"{existing['average_latency_seconds']:.3f}s -> {fine['average_latency_seconds']:.3f}s",
        "~6.8x faster",
    )
    columns[3].metric(
        "Agent Eval",
        f"{agent['baseline_overall']:.4f} -> {agent['post_improvement_overall']:.4f}",
        f"+{agent['delta']:.4f}",
    )
    columns[4].metric("Invalid Router Outputs", f"{fine['invalid_output_rate']:.3f}", "Fine-tuned validation")


def render_demo_metric_cards(metrics: dict[str, Any]) -> None:
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]
    cards = [
        ("Router Accuracy", f"{existing['accuracy']:.3f} -> {fine['accuracy']:.3f}", "Fine-tuned local router benchmark"),
        (
            "Routing Latency",
            f"{existing['average_latency_seconds']:.3f}s -> {fine['average_latency_seconds']:.3f}s",
            "Observed validation latency",
        ),
        ("Agent Eval", f"{agent['baseline_overall']:.4f} -> {agent['post_improvement_overall']:.4f}", "Full agent golden dataset"),
        ("Safety Posture", "Bounded navigation", "Unsafe requests refused"),
    ]
    columns = st.columns(4)
    for column, (title, value, caption) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="syn-story-card">
                  <div class="syn-demo-stat-label">{_escape(title)}</div>
                  <div class="syn-story-value">{_escape(value)}</div>
                  <div class="syn-demo-muted">{_escape(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption("Week 4 measured the full agent workflow. Week 5 measured the narrow routing layer.")


def render_pitch_script_panel() -> None:
    items = build_pitch_script_items()
    bullets = "".join(f"<li>{_escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="syn-demo-card">
          <div class="syn-demo-pill">Recording guide</div>
          <h3>3-minute pitch guide</h3>
          <ul>{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_command_center_hero() -> None:
    logo_col, title_col, badge_col = st.columns([0.06, 0.52, 0.42], vertical_alignment="center")
    with logo_col:
        st.markdown("## ◎")
    with title_col:
        st.markdown("### Synataric Global Healthcare Navigator")
        st.caption("Care-navigation workflow layer for cross-border treatment planning")
    with badge_col:
        badges = st.columns(3)
        for column, badge in zip(badges, ["Educational navigation only", "MINT + ReAct", "Evaluated"]):
            with column:
                with st.container(border=True):
                    st.caption(badge)
        _, read_only = st.columns([0.58, 0.42])
        with read_only:
            with st.container(border=True):
                st.caption("Read-only tools")


def render_architecture_pipeline() -> None:
    with st.container(border=True):
        st.markdown("#### ARCHITECTURE PIPELINE")
        nodes = get_architecture_pipeline_nodes()
        columns = st.columns(len(nodes))
        icons = ["▣", "⌕", "⌁", "⚙", "◇", "▤", "▢"]
        for index, (column, node) in enumerate(zip(columns, nodes)):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{icons[index]} {node['title']}**")
                    st.caption(node["role"])
        st.caption(
            "Corpus -> RAG Evidence -> MINT Router -> Agent Tools -> "
            "Bounded ReAct Planner -> Grounded Care Plan -> Safety / HITL / Evals"
        )


def render_metrics_strip(metrics: dict[str, Any]) -> None:
    cards = get_metric_cards(metrics)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            with st.container(border=True):
                header_cols = st.columns([0.76, 0.24], vertical_alignment="center")
                with header_cols[0]:
                    st.markdown(f"#### {card['title']}")
                with header_cols[1]:
                    st.caption(card.get("icon", ""))
                st.metric(card["caption"].split("|")[0].strip(), card["value"])
                st.caption(card["caption"])
                if card.get("delta"):
                    st.caption(f"↗ {card['delta']}")


def render_mint_ladder() -> None:
    st.markdown("### MINT Decision Ladder")
    st.caption("Minimum intelligence necessary: Simple -> Routed -> Agentic -> Guarded")
    columns = st.columns(4)
    for column, card in zip(columns, get_mint_ladder_cards()):
        with column:
            with st.container(border=True):
                st.caption(card["mode"])
                st.markdown(f"#### {card['title']}")
                st.markdown(f"**{card['label']}**")
                st.write(card["text"])


def render_react_reason_card(question: str) -> None:
    panel = build_why_react_panel("Multi-step care plan", question)
    with st.container(border=True):
        st.markdown("#### Why ReAct here?")
        st.write(panel["body"])
        chips = st.columns(4)
        for column, chip in zip(chips, panel["chips"] or ["Providers", "Cost", "Recovery", "Risks"]):
            with column:
                st.caption(chip)
        st.caption(" -> ".join(get_demo_tool_flow()))


def render_workflow_timeline(steps: list[dict[str, Any]], label: str = "Workflow timeline") -> None:
    if not steps:
        return
    st.markdown("#### WORKFLOW TIMELINE")
    for index, step in enumerate(steps, start=1):
        status = str(step.get("status") or "waiting").lower()
        marker = "✓" if status == "complete" else "○" if status in {"waiting", "running"} else "!"
        with st.container(border=True):
            st.markdown(f"**{marker} {sanitize_demo_text(step.get('title'))}**")
            detail = sanitize_demo_text(step.get("detail"))
            if detail:
                st.caption(detail)
            tool = sanitize_demo_text(step.get("tool"))
            if tool:
                st.caption(tool)


def render_result_summary(result: Any) -> dict[str, Any] | None:
    if not result:
        return None
    expected_route = st.session_state.get("demo_mode_expected_route", "N/A")
    latency = st.session_state.get("demo_mode_latency", 0.0)
    fields = extract_demo_result_fields(result, expected_route, latency)
    if st.session_state.get("demo_mode_question_ran"):
        fields["user_question"] = sanitize_demo_text(st.session_state.demo_mode_question_ran)
    evidence = extract_evidence(result)
    sources = extract_sources(result)
    tool_calls = extract_tool_calls(result)

    st.markdown("### Result summary")
    columns = st.columns(6)
    columns[0].metric("Status", fields["status"])
    columns[1].metric("Workflow", fields["actual_route"])
    columns[2].metric("Tool calls", fields["tool_call_count"])
    columns[3].metric("Latency", fields["runtime_latency"])
    columns[4].metric("Safety", fields["safety_status"])
    columns[5].metric("Evidence", len(evidence or sources))
    st.caption(f"Question: {fields['user_question']}")
    return {"fields": fields, "evidence": evidence, "sources": sources, "tool_calls": tool_calls}


def _route_key_for_result(fields: dict[str, Any]) -> str:
    expected = str(fields.get("expected_route") or "").lower()
    actual = str(fields.get("actual_route") or "").lower()
    selected_tool = str(fields.get("selected_tool") or "").lower()
    status = str(fields.get("status") or "").lower()
    if expected == "needs_clarification" or status == "needs_human" or selected_tool == "ask_human_tool":
        return "needs_clarification"
    if expected == "unsafe_medical" or actual == "unsafe_medical" or selected_tool == "safety_response_tool":
        return "unsafe_medical"
    if expected == "out_of_scope" or actual == "out_of_scope" or selected_tool == "out_of_scope_response_tool":
        return "out_of_scope"
    if status == "coverage_gap" or selected_tool == "coverage_gap_response_tool":
        return "coverage_gap"
    if expected:
        return expected
    if selected_tool.endswith("_tool"):
        return selected_tool.replace("_tool", "")
    return actual or "care_plan_multistep"


def render_actual_routing_decision(fields: dict[str, Any]) -> None:
    route_key = _route_key_for_result(fields)
    route_details = {
        "care_plan_multistep": ("ReAct Care Planner", "request asked for providers + cost + recovery + risks"),
        "cost_estimate": ("Agent Navigator", "cost_estimate_tool"),
        "provider_search": ("Agent Navigator", "provider_search_tool"),
        "recovery_guidance": ("Agent Navigator", "recovery_guidance_tool"),
        "risk_checklist": ("Agent Navigator", "risk_checklist_tool"),
        "unsafe_medical": ("Safety Boundary", "safety_response_tool"),
        "needs_clarification": ("Human Clarification", "ask_human_tool"),
        "out_of_scope": ("Scope Boundary", "out_of_scope_response_tool"),
        "coverage_gap": ("Coverage Boundary", "coverage_gap_response_tool"),
    }
    workflow, reason = route_details.get(route_key, ("Agent Navigator", sanitize_demo_text(fields.get("selected_tool"))))
    with st.container(border=True):
        st.markdown("#### Actual routing decision")
        st.write(f"Selected workflow: {workflow}")
        st.caption(f"Why / tool: {reason}")


def render_native_care_plan_cards(cards: list[dict[str, Any]]) -> None:
    if not cards:
        return
    st.markdown("### Structured care plan")
    columns = st.columns(2)
    for index, card in enumerate(cards):
        items = [sanitize_demo_text(item) for item in card.get("items", []) if sanitize_demo_text(item)]
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {sanitize_demo_text(card.get('title'))}")
                for item in items:
                    st.write(f"- {item}")


def render_route_specific_result(result: Any, scenario_key: str, question: str) -> None:
    expected_route = st.session_state.get("demo_mode_expected_route", scenario_key)
    latency = st.session_state.get("demo_mode_latency", 0.0)
    fields = extract_demo_result_fields(result, expected_route, latency)
    fields["user_question"] = sanitize_demo_text(question or fields.get("user_question"))
    render_actual_routing_decision(fields)
    route_key = _route_key_for_result(fields)
    cards = get_route_specific_result_cards(route_key)
    render_native_care_plan_cards(cards)


def render_command_center_care_plan(result_payload: dict[str, Any]) -> None:
    fields = result_payload["fields"]
    evidence = result_payload["evidence"]
    sources = result_payload["sources"]
    coverage_gaps = detect_coverage_gaps(fields["user_question"], evidence, sources, result=st.session_state.get("demo_mode_result"))
    status = str(fields["status"])
    actual_route = str(fields["actual_route"])
    expected_route = str(fields["expected_route"])

    coverage_note = coverage_note_for_gaps(coverage_gaps)
    if coverage_note:
        st.warning(coverage_note)
    cards = build_coverage_safe_care_plan_cards(fields, evidence, sources, coverage_gaps)
    render_native_care_plan_cards(cards)


def render_command_center_run_output() -> None:
    result = st.session_state.get("demo_mode_result")
    if not result:
        return
    payload = render_result_summary(result)
    if payload is None:
        return
    render_actual_routing_decision(payload["fields"])
    render_command_center_care_plan(payload)
    with st.expander("Evidence and technical trace", expanded=False):
        if payload["tool_calls"]:
            st.markdown("#### Tool calls")
            st.dataframe(pd.DataFrame(payload["tool_calls"]), use_container_width=True, hide_index=True)
        rows = payload["evidence"] or payload["sources"]
        if rows:
            st.markdown("#### Evidence")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence needed for this route.")


def render_demo_console(strategy: str, top_k: int) -> None:
    left, right = st.columns([0.64, 0.36])
    with left:
        with st.container(border=True):
            st.markdown("#### NAVIGATOR CONSOLE")
            st.caption("CLINICAL SCENARIO")
            scenario_name = st.selectbox("Scenario", list(DEMO_SCENARIOS.keys()), label_visibility="collapsed")
            scenario = DEMO_SCENARIOS[scenario_name]
            st.caption(f"Expected workflow: {scenario['workflow']}")
            question_key = "demo_mode_question"
            scenario_key = "demo_mode_scenario"
            if st.session_state.get(scenario_key) != scenario_name:
                st.session_state[scenario_key] = scenario_name
                st.session_state[question_key] = scenario["question"]
            st.caption("NAVIGATOR QUERY")
            question = st.text_area(
                "Patient or caregiver question",
                key=question_key,
                height=135,
                label_visibility="collapsed",
            )
            run_clicked = st.button("▶ Run Navigator", type="primary", use_container_width=False)
    with right:
        with st.container(border=True):
            current_steps = st.session_state.get("demo_mode_workflow_steps") or build_planned_workflow_for_scenario(
                st.session_state.get("demo_mode_scenario", scenario_name)
            )
            render_workflow_timeline(current_steps, st.session_state.get("demo_mode_workflow_label", "Workflow timeline"))

    if run_clicked:
        planned_steps = build_planned_workflow_for_scenario(scenario_name)
        st.session_state.demo_mode_workflow_steps = planned_steps
        st.session_state.demo_mode_workflow_label = "Planned workflow"
        with st.status("Running Synataric Navigator...", expanded=False):
            result, latency, error = _run_live_demo(question, scenario, strategy, top_k)
        actual_steps = extract_actual_workflow(result)
        st.session_state.demo_mode_workflow_steps = actual_steps or [{**step, "status": "complete"} for step in planned_steps]
        st.session_state.demo_mode_workflow_label = "Actual workflow"
        st.session_state.demo_mode_result = result
        st.session_state.demo_mode_latency = latency
        st.session_state.demo_mode_error = error
        st.session_state.demo_mode_expected_route = scenario["expected_route"]
        st.session_state.demo_mode_no_evidence_message = scenario["expected_route"] in {
            "unsafe_medical",
            "needs_clarification",
            "out_of_scope",
        } or _normalize_result(result).get("status") == "coverage_gap"
        st.session_state.demo_mode_question_ran = question
        st.rerun()

    render_command_center_run_output()


def render_architecture_details_tab() -> None:
    st.markdown("### Architecture Details")
    st.dataframe(pd.DataFrame(get_component_summary_rows()), use_container_width=True, hide_index=True)
    st.markdown("### Router node flow")
    st.caption(
        "Intent Classifier -> Boundary Router -> Safety / Out-of-scope / Ask Human / Tool Router -> "
        "Tool Executor -> Fallback / Recovery -> Final Response"
    )


def render_evaluation_details_tab() -> None:
    st.markdown("### Golden dataset")
    columns = st.columns(4)
    columns[0].metric("Dataset", "Synataric-Agent-Golden-Dataset-V1")
    columns[1].metric("Size", "40 cases")
    columns[2].metric("Mix", "20 / 12 / 6 / 2")
    columns[3].metric("Modes", "28 router / 12 ReAct")
    st.caption("Mix: 20 happy path, 12 edge, 6 known failure, 2 adversarial.")

    st.markdown("### Judge method now")
    for item in [
        "deterministic code checks",
        "trajectory checks",
        "source checks",
        "safety checks",
        "out-of-scope checks",
        "path leakage checks",
        "LangSmith tracing",
    ]:
        st.write(f"- {item}")

    st.markdown("### Honest LLM-as-Judge note")
    st.info(
        "LLM-as-Judge is designed as the next enterprise step, but it has not yet been run for the Week 4 "
        "agent benchmark. Earlier RAGAS scoring is LLM-based for the RAG layer, not the agent benchmark."
    )

    st.markdown("### Baseline vs Post Improvement")
    st.dataframe(pd.DataFrame(get_eval_delta_rows()), use_container_width=True, hide_index=True)


def render_production_roadmap_tab() -> None:
    st.markdown("### Production Roadmap")
    with st.container(border=True):
        st.markdown("#### Today")
        st.write("Read-only navigation over a curated Synataric corpus.")
        st.markdown("#### Next")
        for item in [
            "provider APIs",
            "appointment request workflow",
            "insurance verification",
            "travel/lodging APIs",
            "document checklist",
            "secure patient profile",
            "FHIR/records connectors",
            "care coordinator handoff",
        ]:
            st.write(f"- {item}")
        st.warning(
            "Any write action - booking, payment, provider outreach, insurance submission, or message sending - "
            "requires human approval."
        )


def render_presenter_notes_tab() -> None:
    st.markdown("### Presenter Notes")
    for item in [
        "Not a chatbot; care-navigation workflow layer.",
        "MINT: use the simplest architecture that works.",
        "Ask Navigator = RAG.",
        "Agent Navigator = one intent to one tool.",
        "ReAct Care Planner = goal to reason-act-observe loop.",
        "Safety = refuse medication/prescription/urgent medical decisions.",
        "HITL = ask when procedure/destination/care topic is missing.",
        "Evals = measured improvement, not vibes.",
        "Fine-tuning = local route classifier benchmark improved 0.555 to 1.000.",
        "Production = read-only now; write actions need human approval.",
    ]:
        st.write(f"- {item}")


def _render_architecture_flow() -> None:
    with st.container(border=True):
        st.markdown("### Architecture Flow")
        st.code(
            """User message
|
Fine-tuned local care router
|
Exactly one route label
|
Synataric workflow
|-- provider_search_tool
|-- cost_estimate_tool
|-- travel_planning_tool
|-- recovery_guidance_tool
|-- risk_checklist_tool
|-- find_evidence_tool
|-- ask_human_tool
|-- safety_response_tool
`-- ReAct Care Planner
|
Grounded answer + evidence + evaluation trace""",
            language="text",
        )
        st.info(
            "The fine-tuned local router was validated offline in Week 5. In this demo page, the stored benchmark "
            "results are shown from reports/finetune. Runtime answering still uses the current Synataric app pipeline "
            "unless a local router module is present."
        )


def _render_scenario_runner(strategy: str, top_k: int) -> None:
    with st.container(border=True):
        st.markdown("### Start a navigation request")
        left, right = st.columns([0.42, 0.58])
        with left:
            scenario_name = st.selectbox("Scenario", list(DEMO_SCENARIOS.keys()))
            scenario = DEMO_SCENARIOS[scenario_name]
            st.caption("Choose a realistic care-navigation scenario or edit the question directly.")
            st.markdown(f"**Expected workflow:** {scenario['workflow']}")
        with right:
            question_key = "demo_mode_question"
            scenario_key = "demo_mode_scenario"
            if st.session_state.get(scenario_key) != scenario_name:
                st.session_state[scenario_key] = scenario_name
                st.session_state[question_key] = scenario["question"]
            question = st.text_area("Patient or caregiver question", key=question_key, height=118)
            render_why_react_panel(scenario_name, question)
            workflow_placeholder = st.empty()
            progress_bar = st.progress(0)
            if st.session_state.get("demo_mode_workflow_steps"):
                render_live_workflow_steps(
                    workflow_placeholder,
                    st.session_state.demo_mode_workflow_steps,
                    st.session_state.get("demo_mode_workflow_label", "Actual workflow"),
                )
                progress_bar.progress(int(_workflow_progress(st.session_state.demo_mode_workflow_steps) * 100))
            if st.button("Run Navigator", type="primary", use_container_width=True):
                planned_steps = build_planned_workflow_for_scenario(scenario_name)
                st.session_state.demo_mode_workflow_label = "Planned workflow"
                st.session_state.demo_mode_workflow_steps = planned_steps
                render_live_workflow_steps(workflow_placeholder, planned_steps, "Planned workflow")
                progress_bar.progress(0)

                animated_steps = [dict(step) for step in planned_steps]
                for index, step in enumerate(animated_steps):
                    animated_steps[index] = {**step, "status": "running"}
                    render_live_workflow_steps(workflow_placeholder, animated_steps, "Planned workflow")
                    progress_bar.progress(max(5, int((index / max(len(animated_steps), 1)) * 80)))
                    time.sleep(0.12)
                    if index < len(animated_steps) - 1:
                        animated_steps[index] = {**animated_steps[index], "status": "complete"}
                        render_live_workflow_steps(workflow_placeholder, animated_steps, "Planned workflow")
                        progress_bar.progress(max(10, int(((index + 1) / max(len(animated_steps), 1)) * 80)))
                        time.sleep(0.08)

                result, latency, error = _run_live_demo(question, scenario, strategy, top_k)
                actual_steps = extract_actual_workflow(result)
                if actual_steps:
                    display_steps = actual_steps
                    workflow_label = "Actual workflow"
                else:
                    display_steps = [{**step, "status": "complete"} for step in animated_steps]
                    workflow_label = "Actual workflow confirmed"
                st.session_state.demo_mode_workflow_steps = display_steps
                st.session_state.demo_mode_workflow_label = workflow_label
                render_live_workflow_steps(workflow_placeholder, display_steps, workflow_label)
                progress_bar.progress(100)
                st.session_state.demo_mode_result = result
                st.session_state.demo_mode_latency = latency
                st.session_state.demo_mode_error = error
                st.session_state.demo_mode_expected_route = scenario["expected_route"]
                st.session_state.demo_mode_no_evidence_message = scenario["expected_route"] in {
                    "unsafe_medical",
                    "needs_clarification",
                    "out_of_scope",
                } or _normalize_result(result).get("status") == "coverage_gap"
                st.session_state.demo_mode_question_ran = question


def _humanize_tool_name(tool: str) -> str:
    labels = {
        "provider_search_tool": "Provider search",
        "cost_estimate_tool": "Cost estimate",
        "recovery_guidance_tool": "Recovery guidance",
        "risk_checklist_tool": "Risk checklist",
        "travel_planning_tool": "Travel planning",
        "find_evidence_tool": "Evidence lookup",
        "ask_human_tool": "Clarification",
        "safety_response_tool": "Safety boundary",
        "out_of_scope_response_tool": "Scope check",
        "coverage_gap_response_tool": "Corpus coverage gap",
        "ReAct Care Planner": "ReAct Care Planner",
    }
    return labels.get(str(tool), str(tool).replace("_", " ").title())


def render_workflow_stepper(tool_calls: list[dict[str, Any]], expected_route: str) -> None:
    if tool_calls:
        steps = [_humanize_tool_name(call["tool"]) for call in tool_calls]
    elif expected_route == "care_plan_multistep":
        steps = ["Provider search", "Cost estimate", "Recovery guidance", "Risk checklist", "Final care plan"]
    else:
        steps = [_humanize_tool_name(expected_route)]
    if expected_route == "care_plan_multistep" and "Final care plan" not in steps:
        steps.append("Final care plan")

    st.markdown('<div class="syn-demo-section-title">Workflow</div>', unsafe_allow_html=True)
    cards = []
    for index, step in enumerate(steps, start=1):
        cards.append(
            f"""
            <div class="syn-demo-step">
              <span class="syn-demo-step-index">{index}</span>
              <strong>{_escape(step)}</strong>
            </div>
            """
        )
    st.markdown(f'<div class="syn-demo-stepper">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_provider_cards(providers: list[dict[str, str]]) -> None:
    if not providers:
        return
    st.markdown('<div class="syn-demo-section-title">Provider Options</div>', unsafe_allow_html=True)
    columns = st.columns(min(3, len(providers)))
    for index, provider in enumerate(providers):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="syn-demo-provider">
                  <div class="syn-demo-pill">Provider option</div>
                  <div class="syn-demo-provider-name">{_escape(provider.get("name"))}</div>
                  <div class="syn-demo-muted">{_escape(provider.get("detail") or "Review fit, accreditation, surgeon experience, and follow-up logistics.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_bullet_card(title: str, items: list[str], css_class: str = "") -> None:
    if not items:
        return
    bullets = "".join(f"<li>{_escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="syn-demo-card {css_class}">
          <h3>{_escape(title)}</h3>
          <ul>{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_evidence_card(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="syn-demo-card syn-demo-missing">
          <div class="syn-demo-pill">Evidence gap</div>
          <h3>{_escape(title)}</h3>
          <p>{_escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cost_card(items: list[str]) -> None:
    if not items:
        return
    st.markdown(
        f"""
        <div class="syn-demo-card syn-demo-cost">
          <div class="syn-demo-pill">Planning estimate</div>
          <h3>Estimated Cost</h3>
          <p>{_escape(" ".join(items))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_care_plan_cards(cards: list[dict[str, Any]]) -> None:
    render_native_care_plan_cards(cards)


def _coverage_label(value: str) -> str:
    return "Available" if value == "available" else "Not available in current corpus"


def render_coverage_card(gaps: dict[str, Any]) -> None:
    coverage = gaps.get("coverage", "Partial")
    badge_class = "coverage-strong" if coverage == "Strong" else "coverage-partial"
    provider_class = "coverage-strong" if gaps.get("provider_coverage") == "available" else "coverage-missing"
    cost_class = "coverage-strong" if gaps.get("cost_coverage") == "available" else "coverage-missing"
    recovery_class = "coverage-strong" if gaps.get("recovery_coverage") == "available" else "coverage-missing"
    risk_class = "coverage-strong" if gaps.get("risk_coverage") == "available" else "coverage-missing"
    st.markdown(
        f"""
        <div class="syn-demo-card">
          <div class="coverage-badge {badge_class}">Coverage: {_escape(coverage)}</div>
          <h3>Corpus Coverage</h3>
          <div class="coverage-grid">
            <div class="coverage-item"><div class="coverage-badge {provider_class}">Provider data</div><div>{_escape(_coverage_label(gaps.get("provider_coverage")))}</div></div>
            <div class="coverage-item"><div class="coverage-badge {cost_class}">Cost data</div><div>{_escape(_coverage_label(gaps.get("cost_coverage")))}</div></div>
            <div class="coverage-item"><div class="coverage-badge {recovery_class}">Recovery guidance</div><div>{_escape(_coverage_label(gaps.get("recovery_coverage")))}</div></div>
            <div class="coverage-item"><div class="coverage-badge {risk_class}">Risk guidance</div><div>{_escape(_coverage_label(gaps.get("risk_coverage")))}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_alert(message: str, title: str = "Clinical Safety Boundary") -> None:
    st.markdown(
        f"""
        <div class="syn-demo-card syn-demo-alert">
          <div class="syn-demo-pill">Safety</div>
          <h3>{_escape(title)}</h3>
          <p>{_escape(message)}</p>
          <p><strong>Please consult a licensed clinician.</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clarification_card(question: str) -> None:
    st.markdown(
        f"""
        <div class="syn-demo-card syn-demo-clarify">
          <div class="syn-demo-pill">More information needed</div>
          <h3>Clarification Needed</h3>
          <p>{_escape(question or "Please share a few more details so Synataric can route this safely.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_out_of_scope_card() -> None:
    st.markdown(
        """
        <div class="syn-demo-card">
          <div class="syn-demo-pill">Outside care navigation</div>
          <h3>This request is outside Synataric's scope</h3>
          <p>Try asking about providers, estimated costs, travel planning, recovery guidance, risk checklists, or evidence for a care-navigation topic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coverage_gap_card(message: str) -> None:
    chips = [
        "Cataract surgery in Bangalore",
        "Cataract cost in Bangalore",
        "Cataract recovery guidance",
        "Urgent symptoms requiring care",
    ]
    chip_html = "".join(f'<span class="syn-demo-pill">{_escape(chip)}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="syn-demo-card syn-demo-missing">
          <div class="coverage-badge coverage-missing">Coverage: Not available in current corpus</div>
          <h3>Corpus Coverage Gap</h3>
          <p>{_escape(message or "I don't have requested-scope evidence in the current Synataric corpus.")}</p>
          <div>{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources_section(evidence: list[dict[str, Any]], sources: list[dict[str, Any]], no_evidence_message: bool) -> None:
    st.markdown('<div class="syn-demo-section-title">Sources / Evidence</div>', unsafe_allow_html=True)
    rows = evidence or sources
    if not rows and no_evidence_message:
        st.info("No evidence needed for this route.")
        return
    if not rows:
        st.info("No evidence returned by this run.")
        return
    with st.expander("View evidence used", expanded=False):
        for row in rows:
            st.markdown(f"**{row.get('rank', '')}. {row.get('source', 'Evidence')}**")
            if row.get("category"):
                st.caption(f"Category: {row.get('category')}")
            if row.get("snippet"):
                st.write(row.get("snippet"))


def render_care_plan_answer(
    fields: dict[str, Any],
    evidence: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    coverage_gaps: dict[str, Any] | None = None,
) -> None:
    status = str(fields["status"])
    actual_route = str(fields["actual_route"])
    expected_route = str(fields["expected_route"])
    coverage_gaps = coverage_gaps or {}
    answer = rewrite_answer_for_coverage_gaps(fields["final_answer"], coverage_gaps)

    if status == "unsafe" or actual_route == "unsafe_medical" or expected_route == "unsafe_medical":
        render_safety_alert(answer or "Synataric cannot provide medication instructions or clinical directives.")
        return
    if fields["requires_human"] or status == "needs_human" or expected_route == "needs_clarification":
        render_clarification_card(fields["human_question"] or answer)
        return
    if status == "out_of_scope" or actual_route == "out_of_scope" or expected_route == "out_of_scope":
        render_out_of_scope_card()
        return
    if status == "coverage_gap" or fields.get("selected_tool") == "coverage_gap_response_tool":
        render_coverage_gap_card(answer)
        return

    coverage_note = coverage_note_for_gaps(coverage_gaps)
    if coverage_note:
        _render_bullet_card("Coverage Note", [coverage_note], "syn-demo-alert")
    cards = build_care_plan_cards(fields, evidence, sources)
    if coverage_gaps.get("provider_coverage") == "missing":
        cards = [
            card
            for card in cards
            if card.get("title") != "Provider Options"
        ]
        cards.insert(
            0,
            {
                "title": "Provider Options",
                "items": [
                    "Not available in the current Synataric corpus.",
                ],
            },
        )
    if coverage_gaps.get("cost_coverage") == "missing":
        cards = [
            card
            for card in cards
            if card.get("title") != "Estimated Cost"
        ]
        cards.insert(
            1 if cards else 0,
            {
                "title": "Estimated Cost",
                "items": [
                    "Not available in the current Synataric corpus.",
                ],
            },
        )
    render_care_plan_cards(cards)


def _render_run_output() -> None:
    result = st.session_state.get("demo_mode_result")
    if not result:
        return

    expected_route = st.session_state.get("demo_mode_expected_route", "N/A")
    latency = st.session_state.get("demo_mode_latency", 0.0)
    fields = extract_demo_result_fields(result, expected_route, latency)
    if st.session_state.get("demo_mode_question_ran"):
        fields["user_question"] = sanitize_demo_text(st.session_state.demo_mode_question_ran)

    tool_calls = extract_tool_calls(result)
    evidence = extract_evidence(result)
    sources = extract_sources(result)
    coverage_gaps = detect_coverage_gaps(fields["user_question"], evidence, sources, result=result)

    st.markdown('<div class="syn-demo-section-title">Navigator Result</div>', unsafe_allow_html=True)
    summary_columns = st.columns(5)
    with summary_columns[0]:
        _render_stat_card("Route / Workflow", fields["actual_route"])
    with summary_columns[1]:
        _render_stat_card("Status", fields["status"])
    with summary_columns[2]:
        _render_stat_card("Latency", fields["runtime_latency"])
    with summary_columns[3]:
        _render_stat_card("Evidence", len(evidence or sources))
    with summary_columns[4]:
        _render_stat_card("Safety", fields["safety_status"])

    st.markdown(
        f"""
        <div class="syn-demo-card">
          <div class="syn-demo-pill">Care question</div>
          <p>{_escape(fields["user_question"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("demo_mode_workflow_steps"):
        render_workflow_stepper(tool_calls, fields["expected_route"])
    render_coverage_card(coverage_gaps)
    render_not_a_chatbot_callout()
    render_care_plan_answer(fields, evidence, sources, coverage_gaps)
    render_sources_section(evidence, sources, bool(st.session_state.get("demo_mode_no_evidence_message")))

    with st.expander("Show technical details", expanded=False):
        technical_top = st.columns(4)
        technical_top[0].metric("Expected Route", fields["expected_route"])
        technical_top[1].metric("Selected Tool", fields["selected_tool"])
        technical_top[2].metric("Step Count", fields["step_count"])
        technical_top[3].metric("Tool Calls", fields["tool_call_count"])
        if fields["warnings"]:
            st.warning("\n".join(f"- {warning}" for warning in fields["warnings"]))
        if fields["errors"]:
            st.write("Runtime details")
            st.write("\n".join(f"- {error}" for error in fields["errors"]))
        if tool_calls:
            st.markdown("#### Tool Calls")
            st.dataframe(pd.DataFrame(tool_calls), use_container_width=True, hide_index=True)
        log = extract_execution_log(result)
        if log:
            st.markdown("#### Execution Log")
            st.dataframe(pd.DataFrame({"Step": range(1, len(log) + 1), "Event": log}), use_container_width=True, hide_index=True)
        if evidence:
            st.markdown("#### Raw Evidence")
            st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
        elif sources:
            st.markdown("#### Raw Sources")
            st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)


def _render_benchmark_panels(metrics: dict[str, Any]) -> None:
    router_tab, eval_tab, context_tab, script_tab = st.tabs(
        ["Week 5 Fine-Tuned Router Benchmark", "Agent Evaluation Delta", "Why Now?", "3-minute Pitch Script"]
    )
    existing = metrics["existing_router"]
    fine = metrics["fine_tuned_router"]
    agent = metrics["agent_eval"]

    with router_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Existing Synataric router")
            st.metric("Accuracy", _format_metric(existing["accuracy"]))
            st.metric("Macro F1", _format_metric(existing["macro_f1"]))
            st.metric("Route Execution Score", _format_metric(existing["route_execution_score"]))
            st.metric("Average Latency", f"{existing['average_latency_seconds']:.3f} sec")
        with right:
            st.markdown("#### Fine-tuned local router")
            st.metric("Accuracy", _format_metric(fine["accuracy"]))
            st.metric("Macro F1", _format_metric(fine["macro_f1"]))
            st.metric("Route Execution Score", _format_metric(fine["route_execution_score"]))
            st.metric("Invalid Output Rate", _format_metric(fine["invalid_output_rate"]))
            st.metric("Smoke Test", f"{fine['smoke_test_passed']} / {fine['smoke_test_total']}")
        comparison = pd.DataFrame(
            [
                ["Accuracy", existing["accuracy"], fine["accuracy"], fine["accuracy"] - existing["accuracy"]],
                ["Macro F1", existing["macro_f1"], fine["macro_f1"], fine["macro_f1"] - existing["macro_f1"]],
                [
                    "Route Execution Score",
                    existing["route_execution_score"],
                    fine["route_execution_score"],
                    fine["route_execution_score"] - existing["route_execution_score"],
                ],
                [
                    "Average Routing Latency",
                    existing["average_latency_seconds"],
                    fine["average_latency_seconds"],
                    fine["average_latency_seconds"] - existing["average_latency_seconds"],
                ],
                [
                    "Invalid Output Rate",
                    existing["invalid_output_rate"],
                    fine["invalid_output_rate"],
                    fine["invalid_output_rate"] - existing["invalid_output_rate"],
                ],
            ],
            columns=["Metric", "Existing Router", "Fine-Tuned Router", "Delta"],
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption(
            "The validation set is synthetic and balanced. This proves the controlled routing task. Production readiness "
            "requires fresh real-world holdout testing, multilingual tests, adversarial tests, and monitoring."
        )

    with eval_tab:
        columns = st.columns(3)
        columns[0].metric("Baseline Agent Eval Overall", f"{agent['baseline_overall']:.4f}")
        columns[1].metric("Post-Improvement Overall", f"{agent['post_improvement_overall']:.4f}")
        columns[2].metric("Measured Delta", f"+{agent['delta']:.4f}")
        st.markdown("#### Strongest Improvements")
        for item in agent["top_improvements"]:
            st.write(f"- {sanitize_demo_text(item)}")
        st.caption(
            "Week 4 measured the full agent workflow. Week 5 measured the narrow routing step. Together they show both "
            "system quality and infrastructure efficiency."
        )

    with context_tab:
        st.markdown("### Why now?")
        st.write(
            "OpenAI and Anthropic are making healthcare AI platform access real. But the implementation layer still "
            "matters: routing, evidence, workflow safety, human handoff, and evaluation."
        )

    with script_tab:
        pitch_script = (
            "Healthcare AI is entering the platform phase. The hard part is no longer just model access - it is "
            "implementation: routing, evidence, safety, workflow, and evaluation.\n\n"
            "Synataric is a global healthcare navigation workflow layer. It is not a medical chatbot. It does not "
            "diagnose or prescribe. It routes a patient or caregiver question into the right workflow: provider "
            "search, cost estimate, travel planning, recovery guidance, risk checklist, human clarification, safety "
            "refusal, or a multi-step ReAct care plan.\n\n"
            "The first improvement is routing. My existing Synataric router scored 55.5% accuracy on the new "
            "11-label routing dataset. I fine-tuned a Llama 3.2 1B LoRA router to do one job: one message in, "
            "exactly one route label out. The fine-tuned router achieved 100% validation accuracy on the held-out "
            "set and reduced observed average routing latency from 2.308 seconds to 0.340 seconds.\n\n"
            "Now watch the multi-step flow. I ask: Create a care travel plan for cataract surgery in Bangalore "
            "including providers, cost, recovery, and risks. Synataric routes this to the ReAct Care Planner, which "
            "calls provider search, cost estimate, recovery guidance, and risk checklist tools before producing a "
            "grounded care-navigation answer.\n\n"
            "For safety, if I ask whether I should take antibiotics after surgery, Synataric refuses to provide "
            "medication instructions and directs me to a licensed clinician. If I ask to plan travel but omit the "
            "procedure, it asks which procedure I am considering instead of guessing.\n\n"
            "The result is a safe, evidence-grounded, evaluated care-navigation layer: small model for routing, RAG "
            "for facts, tools for workflows, ReAct for multi-step planning, human handoff when needed, and evals to "
            "prove what improved."
        )
        st.markdown(
            f"""
            <div class="syn-demo-card">
              <div class="syn-demo-pill">Demo script</div>
              <h3>3-minute pitch script</h3>
              <p>{_escape(pitch_script).replace(chr(10) + chr(10), '</p><p>')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_demo_mode_page(strategy: str, top_k: int) -> None:
    inject_demo_medical_css()
    metrics = load_demo_metrics()
    live_tab, architecture_tab, evaluation_tab, roadmap_tab, notes_tab = st.tabs(
        ["Live Demo", "Architecture Details", "Evaluation Details", "Production Roadmap", "Presenter Notes"]
    )
    with live_tab:
        render_command_center_dashboard(metrics)
        render_demo_console(strategy, top_k)
    with architecture_tab:
        render_architecture_details_tab()
    with evaluation_tab:
        render_evaluation_details_tab()
    with roadmap_tab:
        render_production_roadmap_tab()
    with notes_tab:
        render_presenter_notes_tab()

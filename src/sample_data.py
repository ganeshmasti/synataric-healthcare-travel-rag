import csv
from pathlib import Path


DISCLAIMER = "Illustrative Synataric sample data only. This is not medical advice, diagnosis, or treatment guidance."


FILES = {
    "raw/procedures/cataract_surgery_guide.md": f"""# Cataract Surgery Travel Planning Guide

{DISCLAIMER}

Cataract surgery is commonly planned as an outpatient eye procedure. Patients often ask about evaluation, lens options, surgery day logistics, follow-up visits, and travel timing.

Documented facts:
- A pre-operative eye evaluation usually includes vision testing, eye pressure checks, retinal review when needed, and intraocular lens measurements.
- Common care steps include medical history review, medication review, consent discussion, the procedure visit, and post-operative follow-up.
- Patients are often asked to arrange a companion for surgery day because vision may be blurred and sedating medicines may be used.
- Follow-up is commonly scheduled within the first few days and again later based on the surgeon's plan.

Estimated travel planning considerations:
- Many travelers plan 5 to 10 days near the treating eye center for evaluation, surgery, and early follow-up.
- People with diabetes, glaucoma, retinal disease, or only one functional eye may need additional review and longer local stay.

Questions to ask a licensed clinician:
- What lens options are appropriate for my situation?
- How soon can I fly after surgery?
- What symptoms after surgery should trigger urgent review?
""",
    "raw/procedures/knee_replacement_guide.md": f"""# Knee Replacement Travel Planning Guide

{DISCLAIMER}

Knee replacement is a major orthopedic procedure that may require planning for hospital stay, physiotherapy, mobility aids, wound care, and return travel.

Documented facts:
- Pre-operative work often includes imaging, anesthesia review, medication review, blood tests, and discussion of rehabilitation expectations.
- Hospital stay varies by patient and provider. Some patients need inpatient rehabilitation or structured outpatient physiotherapy.
- Travel planning should consider mobility assistance, wheelchair access, blood clot prevention discussions, and stair-free accommodation.

Estimated recovery planning considerations:
- Travelers often plan several weeks for surgery, early recovery, and initial physiotherapy before longer-distance travel.
- Return-to-work and return-to-driving timing depends on the treating team's assessment and local legal rules.

Questions to ask a licensed clinician:
- What mobility limitations should I expect during the first two weeks?
- What blood clot risk precautions are relevant for my travel itinerary?
- What physiotherapy schedule should be arranged after discharge?
""",
    "raw/procedures/cardiac_bypass_guide.md": f"""# Cardiac Bypass Travel Planning Guide

{DISCLAIMER}

Cardiac bypass surgery, also called coronary artery bypass grafting or CABG, is a complex heart procedure requiring specialist evaluation and careful travel planning.

Documented facts:
- Evaluation may include cardiology consultation, angiography review, echocardiography, medication review, anesthesia assessment, and surgical risk discussion.
- Recovery planning can include intensive care stay, ward recovery, breathing exercises, wound care, cardiac rehabilitation, and follow-up with cardiology.
- Long-distance travel after major cardiac surgery should be discussed directly with the treating cardiac team.

Estimated planning considerations:
- International or domestic medical travel for bypass surgery may require several weeks near the hospital for evaluation, surgery, and early recovery.
- Caregivers should plan for medication management, transport support, and emergency contact pathways.

Urgent symptoms:
- Chest pain, severe breathlessness, fainting, stroke-like symptoms, or heavy bleeding require immediate medical care.

Questions to ask a licensed clinician:
- What is my individualized surgical risk profile?
- How long should I stay near the hospital before return travel?
- What cardiac rehabilitation plan should I follow?
""",
    "raw/hospitals/provider_profiles.md": f"""# Provider Profiles

{DISCLAIMER}

Synataric Navigator stores provider profile notes to help compare logistics and care-navigation questions. These profiles are illustrative.

Bangalore Eye Centre profile:
- Focus areas: cataract evaluation, phacoemulsification, intraocular lens counseling, retinal referral pathways.
- Navigation strengths: airport pickup partners, English and Kannada support, nearby serviced apartments.
- Ask about: lens inventory, retina backup, follow-up schedule, emergency eye contact.

South City Ortho Institute profile:
- Focus areas: joint replacement, sports medicine, physiotherapy coordination.
- Navigation strengths: accessible rooms, physiotherapy desk, wheelchair transport options.
- Ask about: implant choices, rehabilitation schedule, infection prevention policies, return travel timing.

Metro Heart Institute profile:
- Focus areas: cardiology evaluation, bypass surgery, cardiac rehabilitation.
- Navigation strengths: ICU family communication desk, cardiac rehab planning, medication reconciliation support.
- Ask about: surgical risk counseling, expected hospital stay, caregiver accommodation, emergency escalation.
""",
    "raw/risks/travel_medical_risk_checklist.md": f"""# Travel Medical Risk Checklist

{DISCLAIMER}

Use this checklist for care-navigation conversations with licensed professionals.

Before travel:
- Confirm fitness to travel with a licensed clinician.
- Carry prescriptions, allergy list, implant cards, recent test reports, and emergency contacts.
- Ask whether vaccinations, infection precautions, or medication adjustments are needed.
- Confirm travel insurance exclusions and medical evacuation coverage.

During stay:
- Keep provider contact numbers available.
- Use accessible transport when mobility is limited.
- Avoid changing medicines without clinician guidance.
- Seek immediate medical care for chest pain, severe breathlessness, stroke-like symptoms, uncontrolled bleeding, or sudden vision loss.

Before return:
- Ask for discharge summary, operative note when applicable, medication list, follow-up plan, and warning signs.
- Confirm whether flying, long road travel, or train travel is suitable for the specific recovery stage.
""",
    "raw/risks/post_op_recovery_guidelines.md": f"""# Post-operative Recovery Planning Guidelines

{DISCLAIMER}

Recovery planning is procedure-specific and should be confirmed by the treating team.

Documented navigation topics:
- Wound care instructions and dressing schedule.
- Medication list with timing, purpose, and refill plan.
- Follow-up visit dates and telehealth options.
- Red-flag symptoms that need urgent review.
- Mobility support, nutrition, caregiver tasks, and sleep arrangements.

Estimated planning topics:
- Cataract travelers may need early eye follow-up before leaving the city.
- Knee replacement travelers may need physiotherapy access and mobility-friendly lodging.
- Cardiac bypass travelers may need cardiac rehabilitation planning and longer observation near the hospital.
""",
    "raw/policies/synataric_disclaimer_and_safety.md": f"""# Synataric Disclaimer and Safety Policy

{DISCLAIMER}

Synataric Navigator is an educational healthcare navigation assistant. It helps organize questions, logistics, cost estimates, provider profile details, and care pathway information from a curated corpus.

It does not diagnose conditions, prescribe treatment, replace a licensed clinician, or determine whether a procedure is appropriate.

Emergency guidance:
- For chest pain, severe breathlessness, fainting, stroke-like symptoms, heavy bleeding, sudden severe pain, sudden vision loss, or any urgent concern, seek immediate medical care through local emergency services or the nearest emergency department.

Cost guidance:
- Costs in this corpus are illustrative estimates. Actual costs depend on clinician assessment, hospital billing, procedure complexity, implants or lenses, length of stay, complications, currency changes, and insurance terms.
""",
}


CSV_FILES = {
    "raw/hospitals/bangalore_eye_hospitals.csv": [
        ["hospital_name", "city", "focus_area", "navigation_features", "illustrative_notes"],
        ["Bangalore Eye Centre", "Bangalore", "Cataract and retina referral", "Airport pickup partners; multilingual desk; nearby serviced apartments", "Illustrative profile only"],
        ["Narayana Nethra Network", "Bangalore", "Advanced eye care", "Multi-location scheduling; diagnostics coordination", "Illustrative profile only"],
        ["Sankara Eye Services", "Bangalore", "Cataract and community eye care", "Counseling desk; package estimate support", "Illustrative profile only"],
    ],
    "raw/costs/india_procedure_costs.csv": [
        ["procedure", "city", "low_estimate_inr", "high_estimate_inr", "cost_notes"],
        ["Cataract surgery", "Bangalore", "45000", "150000", "Lens choice, diagnostics, surgeon fee, and facility type affect pricing"],
        ["Knee replacement", "Bangalore", "220000", "650000", "Implant choice, room category, hospital stay, and physiotherapy affect pricing"],
        ["Cardiac bypass surgery", "Bangalore", "350000", "900000", "ICU stay, graft complexity, tests, and recovery course affect pricing"],
    ],
    "raw/costs/travel_stay_costs.csv": [
        ["item", "city", "low_estimate_inr", "high_estimate_inr", "planning_notes"],
        ["Serviced apartment per night", "Bangalore", "2500", "8500", "Accessible rooms and caregiver space may cost more"],
        ["Local medical transport per trip", "Bangalore", "800", "3500", "Wheelchair van or late-night transport may cost more"],
        ["Caregiver daily meals and local travel", "Bangalore", "1200", "3000", "Varies by location and length of stay"],
        ["Airport transfer", "Bangalore", "1800", "6000", "Distance, vehicle type, and mobility needs affect cost"],
    ],
    "sample_questions.csv": [
        ["question"],
        ["How many days should I plan to stay in Bangalore for cataract surgery?"],
        ["What should a caregiver ask before knee replacement travel?"],
        ["What are illustrative cost ranges for cardiac bypass and recovery logistics?"],
        ["Which urgent symptoms should not be handled by the navigator?"],
    ],
}


def _write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_csv_if_missing(path: Path, rows: list[list[str]]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def create_sample_corpus(data_dir: str | Path | None = None) -> None:
    base = Path(data_dir or Path(__file__).resolve().parents[1] / "data")
    for relative, content in FILES.items():
        _write_text_if_missing(base / relative, content)
    for relative, rows in CSV_FILES.items():
        _write_csv_if_missing(base / relative, rows)

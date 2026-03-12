from models import AIExtraction, Vitals

# Hardcoded Medical Red Flags
RED_FLAGS = ["chest pain", "shortness of breath", "stroke", "bleeding", "unconscious", "numbness", "blur", "slurred"]

def calculate_priority(extraction: AIExtraction, vitals: Vitals = None) -> tuple[int, str, str]:
    """Calculates risk score and generates the Explainability string."""
    base_score = 10
    explanations = ["Base triage initiated (+10)."]

    # 1. Red Flag Overrides
    found_flags = [sym for sym in extraction.extracted_symptoms if any(rf in sym.lower() for rf in RED_FLAGS)]
    if found_flags:
        penalty = 30 + (len(found_flags) * 10)
        base_score += penalty
        explanations.append(f"Critical red flags detected: {', '.join(found_flags)} (+{penalty}).")

    # 2. History / Underlying Conditions
    if extraction.underlying_conditions:
        base_score += 15
        explanations.append(f"Underlying condition risk factor (+15).")

    # 3. Vitals Math (If the frontend passed them)
    if vitals:
        if vitals.heart_rate and (vitals.heart_rate > 120 or vitals.heart_rate < 50):
            base_score += 20
            explanations.append(f"Abnormal HR ({vitals.heart_rate} bpm) (+20).")
        if vitals.temperature and vitals.temperature > 103.0:
            base_score += 15
            explanations.append(f"High Fever (+15).")

    final_score = min(base_score, 100)

    # 4. Routing logic mapping
    if final_score >= 80:
        urgency = "Critical"
    elif final_score >= 50:
        urgency = "Urgent"
    elif final_score >= 30:
        urgency = "Moderate"
    else:
        urgency = "Low"

    explainability_summary = f"Risk Score: {final_score}. Justification: " + " ".join(explanations)

    return final_score, urgency, explainability_summary

def calculate_rescore(current_score: int, wait_time_minutes: int) -> int:
    """Lightweight wait-time penalty math."""
    # Add 3 points for every 10 minutes spent waiting
    bump = (wait_time_minutes // 10) * 3
    return min(current_score + bump, 100)
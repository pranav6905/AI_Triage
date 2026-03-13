from models import AIExtraction, Vitals, PatientContext

def calculate_priority(extraction: AIExtraction, context: PatientContext, vitals: Vitals = None) -> tuple[int, str, str]:
    """Calculates risk score and generates the Explainability string."""
    explanations = []

    # --- PHASE 1: CRITICAL ASSESSMENT ---
    p1_score = 0
    if not context.is_conscious:
        p1_score = 100
        explanations.append("Unconscious (CRITICAL +100)")
    elif context.breathing_difficulty == "severe":
        p1_score = 40
        explanations.append("Severe breathing difficulty (+40)")
    elif context.breathing_difficulty == "mild":
        p1_score = 20
        explanations.append("Mild breathing difficulty (+20)")

    # --- PHASE 2: SYMPTOM ANALYSIS ---
    p2_score = 0
    if extraction.severity == "severe": 
        p2_score = 50
        explanations.append("Severe symptoms (+50)")
    elif extraction.severity == "moderate": 
        p2_score = 30
        explanations.append("Moderate symptoms (+30)")
    else: 
        p2_score = 10

    # --- PHASE 3: PATIENT CONTEXT ---
    p3_score = 0
    if context.age:
        if context.age < 5 or context.age > 75: 
            p3_score += 15
            explanations.append("High-risk age (+15)")
        elif 5 <= context.age <= 18 or 65 <= context.age <= 75: 
            p3_score += 10
            explanations.append("Moderate-risk age (+10)")
    
    for comorb in context.comorbidities:
        c = comorb.lower()
        if "diabetes" in c: p3_score += 10
        if "hypertension" in c or "bp" in c: p3_score += 8
        if "heart" in c or "cardiac" in c: p3_score += 25
    if p3_score > 0:
        explanations.append(f"Comorbidities factor (+{min(p3_score, 40)})")
    p3_score = min(p3_score, 40)

    # --- PHASE 4: VITALS & OBJECTIVE ---
    p4_score = 0
    if vitals:
        if vitals.blood_pressure:
            try:
                sys, dia = map(int, vitals.blood_pressure.split('/'))
                if sys > 180 or dia > 120: p4_score += 20
            except: pass
        if vitals.heart_rate and (vitals.heart_rate > 120 or vitals.heart_rate < 50): 
            p4_score += 15
        if vitals.temperature and vitals.temperature > 40.0: 
            p4_score += 15
        if vitals.o2_sat and vitals.o2_sat < 94: 
            p4_score += 25
            explanations.append("Low O2 Saturation (+25)")
        if vitals.respiratory_rate and vitals.respiratory_rate > 30: 
            p4_score += 20
    if p4_score > 0:
        explanations.append(f"Vitals factor (+{min(p4_score, 50)})")
    p4_score = min(p4_score, 50) 

    # --- PHASE 5: TIMELINE ---
    p5_score = 0
    if extraction.onset_type == "sudden": 
        p5_score += 10
        explanations.append("Sudden onset (+10)")
    if context.recent_trauma_or_surgery:
        p5_score += 15
        explanations.append("Recent trauma/surgery (+15)")
    p5_score = min(p5_score, 20)

    # --- WEIGHTED ALGORITHM ---
    weighted_total = (
        (p1_score * 0.40) + 
        ((p2_score * 2) * 0.30) + 
        ((p3_score * 2.5) * 0.15) + 
        ((p4_score * 2) * 0.10) + 
        ((p5_score * 5) * 0.05)
    )
    final_score = min(int(weighted_total), 100)

    # --- RED FLAG OVERRIDES ---
    if extraction.detected_red_flags and final_score < 80:
        final_score = 90  # Force to Critical
        explanations.append(f"CRITICAL OVERRIDE: AI detected {', '.join(extraction.detected_red_flags)}.")

    # --- PRIORITY ASSIGNMENT ---
    if final_score >= 90: urgency = "Critical"
    elif final_score >= 70: urgency = "High"
    elif final_score >= 50: urgency = "Moderate"
    elif final_score >= 30: urgency = "Low"
    else: urgency = "Routine"

    explainability_summary = f"Total Risk: {final_score}. Breakdown: " + " | ".join(explanations)
    return final_score, urgency, explainability_summary

def calculate_rescore(risk_score: int, wait_time_minutes: int, urgency_level: str) -> tuple[int, str]:
    """Lightweight wait-time penalty math and priority escalation."""
    new_score = risk_score
    
    # Base penalty: Add 3 points for every 10 minutes spent waiting
    new_score += (wait_time_minutes // 10) * 3
    
    # Priority escalation logic (Dynamic adjustment prevents "stuck in queue")
    if wait_time_minutes > 30 and urgency_level == "High":
        new_score = max(new_score, 90) # Bump to Critical
    elif wait_time_minutes > 60 and urgency_level == "Moderate":
        new_score = max(new_score, 70) # Bump to High
        
    new_score = min(new_score, 100)
    
    # Re-evaluate category based on new score
    if new_score >= 90: new_urgency = "Critical"
    elif new_score >= 70: new_urgency = "High"
    elif new_score >= 50: new_urgency = "Moderate"
    elif new_score >= 30: new_urgency = "Low"
    else: new_urgency = "Routine"
    
    return new_score, new_urgency
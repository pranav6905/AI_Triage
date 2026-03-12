from fastapi import FastAPI
from models import (
    TriageRequest, TriageResponse, AIExtraction, 
    BatchRescoreRequest, BatchRescoreResponse, RescoreResult
)
from ai_service import extract_clinical_data
from scoring_engine import calculate_priority, calculate_rescore

app = FastAPI(title="JEEVA Triage AI - HF Serverless")

@app.post("/api/v1/analyze-triage", response_model=TriageResponse)
async def analyze_triage(request: TriageRequest):
    try:
        # Phase 1 & 2: LLM Extraction via Hugging Face
        ai_data = extract_clinical_data(request.raw_text)
        
        # Phase 3 & 4: Python Scoring Math & Explainability
        risk_score, urgency_level, explainability = calculate_priority(ai_data, request.vitals)
        
        # Phase 5: Final Aggregation
        return TriageResponse(
            patient_id=request.patient_id,
            risk_score=risk_score,
            urgency_level=urgency_level,
            department=ai_data.department,
            explainability_summary=explainability,
            ai_analysis=ai_data
        )
        
    except Exception as e:
        # THE FALLBACK: If HF API times out or Llama hallucinates bad JSON
        fallback_data = AIExtraction(
            chief_complaint=request.raw_text,
            extracted_symptoms=["Parse Failed"],
            underlying_conditions=[],
            department="General"
        )
        return TriageResponse(
            patient_id=request.patient_id,
            risk_score=50, # Safe middle-ground score
            urgency_level="Pending Manual Triage",
            department="General",
            explainability_summary=f"AI Engine offline or failed. Awaiting human review. Error log: {str(e)}",
            ai_analysis=fallback_data
        )

@app.post("/api/v1/rescore-batch", response_model=BatchRescoreResponse)
async def rescore_batch_patients(request: BatchRescoreRequest):
    """
    Receives an array of waiting patients, calculates their new wait-time 
    penalty score, and returns the updated array.
    """
    updated_results = []
    
    for patient in request.patients:
        # Calculate the new score using your existing scoring_engine math
        new_score = calculate_rescore(patient.current_score, patient.wait_time_minutes)
        
        # Append the result to our list
        updated_results.append(
            RescoreResult(
                patient_id=patient.patient_id,
                updated_risk_score=new_score,
                message=f"Score bumped to {new_score} (+ {patient.wait_time_minutes} mins waited)"
            )
        )
        
    return BatchRescoreResponse(results=updated_results)
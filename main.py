from fastapi import FastAPI
from models import (
    NextQuestionsRequest, NextQuestionsResponse,
    TriageRequest, TriageResponse, AIExtraction, 
    BatchRescoreRequest, BatchRescoreResponse, RescoreResult
)
from ai_service import extract_clinical_data, generate_next_questions
from scoring_engine import calculate_priority, calculate_rescore

app = FastAPI(title="JEEVA Dynamic Triage AI (Gemini Edition)")

@app.post("/api/v1/chat/next-questions", response_model=NextQuestionsResponse)
async def get_next_questions(request: NextQuestionsRequest):
    """
    Endpoint 1: During the chat loop. 
    Frontend sends the history so far, AI returns the next 1-2 questions to ask.
    """
    questions = generate_next_questions(request.conversation_history)
    return NextQuestionsResponse(questions=questions)

@app.post("/api/v1/analyze-triage", response_model=TriageResponse)
async def analyze_triage(request: TriageRequest):
    """
    Endpoint 2: End of the chat loop.
    Frontend sends the FULL chat history + available departments + context + vitals.
    """
    try:
        # Pass both the history AND the dynamic available departments to the AI
        ai_data = extract_clinical_data(
            history=request.conversation_history, 
            available_departments=request.available_departments
        )
        
        # Phase 1, 3, 4, & Math Evaluation
        risk_score, urgency_level, explainability = calculate_priority(
            extraction=ai_data, 
            context=request.context, 
            vitals=request.vitals
        )
        
        return TriageResponse(
            patient_id=request.patient_id,
            risk_score=risk_score,
            urgency_level=urgency_level,
            department=ai_data.department, 
            explainability_summary=explainability,
            ai_analysis=ai_data
        )
        
    except Exception as e:
        # THE FALLBACK: If API times out or hallucinates
        fallback_data = AIExtraction(
            chief_complaint="Failed to parse conversation",
            extracted_symptoms=[],
            detected_red_flags=[],
            severity="moderate",
            symptom_category="general",
            onset_type="gradual",
            department="General"
        )
        return TriageResponse(
            patient_id=request.patient_id,
            risk_score=50,
            urgency_level="Moderate",
            department="General",
            explainability_summary=f"AI Engine offline or failed. Error log: {str(e)}",
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
        # Now using the strictly standardized names: risk_score and urgency_level
        new_score, new_urgency = calculate_rescore(
            patient.risk_score, 
            patient.wait_time_minutes,
            patient.urgency_level
        )
        
        updated_results.append(
            RescoreResult(
                patient_id=patient.patient_id,
                risk_score=new_score,
                urgency_level=new_urgency,
                message=f"Score bumped to {new_score} (+ wait time rules)"
            )
        )
        
    return BatchRescoreResponse(results=updated_results)
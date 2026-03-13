from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google.cloud import vision
import fitz
from models import (
    NextQuestionsRequest, NextQuestionsResponse,
    TriageRequest, TriageResponse, AIExtraction, 
    BatchRescoreRequest, BatchRescoreResponse, RescoreResult
)
from ai_service import extract_clinical_data, generate_next_questions, summarize_ocr_text
from scoring_engine import calculate_priority, calculate_rescore

app = FastAPI(title="JEEVA Dynamic Triage AI (Gemini Edition)")

# Initialize Google Cloud Vision Client
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Warning: Google Cloud Vision client failed to initialize. Error: {e}")
    vision_client = None

@app.post("/api/v1/chat/next-questions", response_model=NextQuestionsResponse)
async def get_next_questions(request: NextQuestionsRequest):
    """
    Endpoint 1: During the chat loop. 
    Frontend sends the history so far, AI returns the next 1-2 questions to ask.
    """
    questions = generate_next_questions(request.conversation_history)
    return NextQuestionsResponse(questions=questions)

@app.post("/api/v1/analyze-triage", response_model=TriageResponse)
@app.post("/api/v1/analyze-triage", response_model=TriageResponse)
async def analyze_triage(
    payload: str = Form(..., description="""
    **EXPECTED JSON STRUCTURE (Pass as a stringified JSON):**
    ```json
    {
      "patient_id": "string",
      "conversation_history": [
        {"role": "assistant", "content": "string"},
        {"role": "user", "content": "string"}
      ],
      "available_departments": ["Cardiology", "Neurology"],
      "context": {
        "is_conscious": true,
        "breathing_difficulty": "normal",
        "age": 0,
        "comorbidities": ["string"],
        "recent_trauma_or_surgery": false
      },
      "vitals": {
        "heart_rate": 0,
        "blood_pressure": "120/80",
        "temperature": 0,
        "o2_sat": 0,
        "respiratory_rate": 0
      }
    }
    ```
    """), 
    file: UploadFile = File(None, description="Optional: Upload a PDF or Image of medical records")
):
    """
    MULTI-AGENT ENDPOINT:
    Agent 1 (Optional): OCR extracts text from PDF/Image & summarizes history.
    Agent 2: Analyzes conversation + Agent 1's summary to calculate risk scores.
    """
    # Parse the incoming JSON string payload into our Pydantic model
    try:
        request_data = json.loads(payload)
        request = TriageRequest(**request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")

    # ==========================================
    # AGENT 1: Document Extraction & Summarization
    # ==========================================
    if file and vision_client:
        try:
            content = await file.read()
            full_raw_text = ""
            
            # Extract PDF
            if file.filename.lower().endswith(".pdf"):
                doc = fitz.open(stream=content, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150) 
                    image = vision.Image(content=pix.tobytes("png"))
                    response = vision_client.document_text_detection(image=image)
                    if not response.error.message:
                        full_raw_text += response.full_text_annotation.text + "\n\n"
            # Extract Image
            else:
                image = vision.Image(content=content)
                response = vision_client.document_text_detection(image=image)
                if not response.error.message:
                    full_raw_text = response.full_text_annotation.text

            # If text was found, let Gemini (Agent 1) summarize it
            if full_raw_text.strip():
                summary = summarize_ocr_text(full_raw_text)
                # Inject Agent 1's summary directly into the patient context for Agent 2
                request.context.historical_summary = summary
                
        except Exception as e:
            print(f"Warning: Agent 1 (OCR/Summary) failed. Proceeding without document. Error: {e}")

    # ==========================================
    # AGENT 2: Triage & Scoring Engine
    # ==========================================
    try:
        # Agent 2 now receives the chat history AND Agent 1's summary
        ai_data = extract_clinical_data(
            history=request.conversation_history, 
            available_departments=request.available_departments,
            historical_summary=request.context.historical_summary
        )
        
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
            historical_summary=request.context.historical_summary,
            ai_analysis=ai_data
        )
        
    except Exception as e:
        # THE FALLBACK
        fallback_data = AIExtraction(
            chief_complaint="Failed to parse conversation",
            extracted_symptoms=[],
            detected_red_flags=[],
            severity="moderate",
            symptom_category="general",
            onset_type="gradual",
            department="General",
            extracted_comorbidities=[]
        )
        return TriageResponse(
            patient_id=request.patient_id,
            risk_score=50,
            urgency_level="Moderate",
            department="General",
            explainability_summary=f"AI Engine offline or failed. Error log: {str(e)}",
            historical_summary=request.context.historical_summary if hasattr(request, 'context') else None,
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


from pydantic import BaseModel
from typing import List, Optional

# --- CHAT & HISTORY MODELS ---
class Message(BaseModel):
    role: str      # strictly "user" (patient) or "assistant" (AI)
    content: str   # The text of the message

class NextQuestionsRequest(BaseModel):
    conversation_history: List[Message]

class NextQuestionsResponse(BaseModel):
    questions: List[str]

# --- INCOMING TRIAGE REQUESTS ---
class Vitals(BaseModel):
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None  
    temperature: Optional[float] = None   
    o2_sat: Optional[int] = None          
    respiratory_rate: Optional[int] = None

class PatientContext(BaseModel):
    is_conscious: bool = True
    breathing_difficulty: str = "normal"  # "normal", "mild", or "severe"
    age: Optional[int] = None
    comorbidities: List[str] = []         # e.g., ["diabetes", "hypertension"]
    recent_trauma_or_surgery: bool = False
    historical_summary: Optional[str] = None

class TriageRequest(BaseModel):
    patient_id: str
    conversation_history: List[Message]   # Full chat history
    available_departments: List[str]      # Dynamic hospital departments
    context: PatientContext
    vitals: Optional[Vitals] = None

# --- INTERNAL AI STRUCTURES ---
class AIExtraction(BaseModel):
    chief_complaint: str
    extracted_symptoms: List[str]
    detected_red_flags: List[str]         
    severity: str                         
    symptom_category: str                 
    onset_type: str                       
    department: str

# --- OUTGOING RESPONSE (Endpoint 2) ---
class TriageResponse(BaseModel):
    patient_id: str
    risk_score: int 
    urgency_level: str 
    department: str
    explainability_summary: str
    ai_analysis: AIExtraction

# --- BATCH RESCORE (Endpoint 3) ---
class RescoreItem(BaseModel):
    patient_id: str
    risk_score: int               # STANDARD NAME
    urgency_level: str            # STANDARD NAME
    wait_time_minutes: int

class BatchRescoreRequest(BaseModel):
    patients: List[RescoreItem]

class RescoreResult(BaseModel):
    patient_id: str
    risk_score: int               # STANDARD NAME
    urgency_level: str            # STANDARD NAME
    message: str

class BatchRescoreResponse(BaseModel):
    results: List[RescoreResult]
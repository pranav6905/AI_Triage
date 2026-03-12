from pydantic import BaseModel
from typing import List, Optional

# --- INCOMING REQUESTS ---
class Vitals(BaseModel):
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None 
    temperature: Optional[float] = None

class TriageRequest(BaseModel):
    patient_id: str
    raw_text: str 
    vitals: Optional[Vitals] = None

class RescoreRequest(BaseModel):
    patient_id: str
    current_score: int
    wait_time_minutes: int

# --- INTERNAL AI STRUCTURES ---
class AIExtraction(BaseModel):
    chief_complaint: str
    extracted_symptoms: List[str]
    duration_minutes: Optional[int] = None
    underlying_conditions: List[str]
    department: str

# --- OUTGOING RESPONSE ---
class TriageResponse(BaseModel):
    patient_id: str
    risk_score: int 
    urgency_level: str 
    department: str
    explainability_summary: str
    ai_analysis: AIExtraction

# --- BATCH RESCORE REQUEST MODELS ---
class RescoreItem(BaseModel):
    patient_id: str
    current_score: int
    wait_time_minutes: int

class BatchRescoreRequest(BaseModel):
    patients: List[RescoreItem]

# --- BATCH RESCORE RESPONSE MODELS ---
class RescoreResult(BaseModel):
    patient_id: str
    updated_risk_score: int
    message: str

class BatchRescoreResponse(BaseModel):
    results: List[RescoreResult]
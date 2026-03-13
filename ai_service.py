import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from models import AIExtraction, Message
from typing import List

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini 1.5 Flash for speed and native JSON support
model = genai.GenerativeModel('gemini-2.5-flash')

# --- PROMPT 1: Generating the next questions ---
QUESTION_PROMPT = """You are an expert, empathetic ER triage nurse. Read the conversation history between the triage system (assistant) and the patient (user). 
Based on what the patient has said so far, generate 1 or 2 natural, conversational follow-up questions.

RULE 1: Start with basic clarifications (e.g., "Can you describe the pain?", "Exactly where does it hurt?", "Is it sharp or dull?").
RULE 2: Do NOT jump immediately to extreme or scary symptoms unless the patient's condition sounds immediately life-threatening.
RULE 3: Keep it short, simple, and comforting. Do not diagnose.

OUTPUT EXACTLY IN THIS JSON FORMAT:
{
  "questions": ["Question 1?", "Question 2?"]
}"""

# --- UPDATED PROMPT: Strict Plain-Text Multilingual OCR Summarization ---
OCR_SUMMARY_PROMPT = """You are a multilingual clinical data assistant. You are given raw, messy text extracted from a patient's medical document via OCR. 

CRITICAL INSTRUCTIONS:
1. The text may be in English, Hindi, Gujarati, Marathi, or any other language. Understand the context and translate it into ENGLISH.
2. Extract ONLY the relevant medical history, previous diagnoses, chronic conditions, and current medications.
3. Ignore hospital addresses, phone numbers, page numbers, and irrelevant administrative text.

STRICT FORMATTING RULES:
- DO NOT use any Markdown formatting (no asterisks *, no bolding, no bullet points).
- DO NOT use line breaks or newline characters (\n). Output a single, continuous paragraph.
- DO NOT state the language of the original document.
- DO NOT use introductory labels like "Medical Summary:" or "The document is...". Just output the facts.

If no useful medical history is found, output exactly: "No significant medical history found in document."
"""

# --- PROMPT 2: Final Data Extraction (DYNAMIC TEMPLATE) ---
# Notice the double {{ }} for JSON to prevent Python string format errors
EXTRACTION_PROMPT_TEMPLATE = """You are an expert clinical data extraction AI. Read the entire triage conversation history between the patient and the assistant AND the patient's historical medical context (if provided).
Your ONLY job is to extract medical entities from this conversation and output raw, valid JSON.

Extract the following:
1. chief_complaint: A short summary of the main issue.
2. extracted_symptoms: Array of specific symptoms mentioned by the patient.
3. detected_red_flags: Array of critical life-threatening indicators (e.g., "chest pain", "shortness of breath"). Include any severe risks found in their historical context if relevant to the current complaint. If none, output [].
5. symptom_category: e.g., "cardiac", "neurological", "respiratory", "gastrointestinal", "orthopedic", "general".
6. onset_type: strictly "sudden", "gradual", or "chronic".
7. department: Choose strictly from the following available hospital departments: {available_departments}. If none fit perfectly, choose the closest match or the general department.

OUTPUT EXACTLY IN THIS JSON FORMAT:
{{
  "chief_complaint": "string",
  "extracted_symptoms": ["string"],
  "detected_red_flags": ["string"],
  "severity": "string",
  "symptom_category": "string",
  "onset_type": "string",
  "department": "string"
}}"""

def format_history(history: List[Message]) -> str:
    """Helper to convert the message objects into a readable transcript for the LLM."""
    transcript = ""
    for msg in history:
        transcript += f"{msg.role.upper()}: {msg.content}\n"
    return transcript

def generate_next_questions(history: List[Message], patient_language="English") -> List[str]:
    """Generates the next logical questions based on the chat so far."""
    transcript = format_history(history)
    full_prompt = f"{QUESTION_PROMPT}\n\nConversation so far:\n{transcript}"
    
    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3
        )
    )
    
    try:
        parsed_json = json.loads(response.text)
        return parsed_json.get("questions", ["Could you provide more details about how you are feeling?"])
    except:
        return ["Could you describe your symptoms a bit more?"]

def extract_clinical_data(history: List[Message], available_departments: List[str], historical_summary: str = None) -> AIExtraction:
    """Extracts structured data and assigns a department based ONLY on the provided list."""
    transcript = format_history(history)
    
    # Inject the frontend's dynamic departments into the prompt
    departments_str = ", ".join(available_departments)
    formatted_prompt = EXTRACTION_PROMPT_TEMPLATE.format(available_departments=departments_str)

    # NEW: Format the history block
    history_block = f"Historical Medical Context:\n{historical_summary}\n\n" if historical_summary else "Historical Medical Context:\nNone provided.\n\n"
    
    # NEW: Inject it into the full prompt
    full_prompt = f"{formatted_prompt}\n\n{history_block}Conversation to extract from:\n{transcript}"
    
    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    parsed_json = json.loads(response.text)
    return AIExtraction(**parsed_json)

def summarize_ocr_text(raw_text: str) -> str:
    """Passes raw OCR garbage to Gemini to clean up into a neat medical history."""
    full_prompt = f"{OCR_SUMMARY_PROMPT}\n\nRaw OCR Text:\n{raw_text}"
    response = model.generate_content(full_prompt, generation_config=genai.GenerationConfig(temperature=0.1))
    return response.text.strip()
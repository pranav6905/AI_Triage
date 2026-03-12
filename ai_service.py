import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from models import AIExtraction

load_dotenv()

# Initialize the Hugging Face Client targeting Llama 3
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

SYSTEM_PROMPT = """You are an expert clinical data extraction engine. Your ONLY job is to extract medical entities and output raw, valid JSON. Do NOT include markdown blocks, greetings, or conversational text.

Extract the following from the narrative:
1. chief_complaint
2. extracted_symptoms (array of strings)
3. duration_minutes (integer estimate, or null)
4. underlying_conditions (array of strings)
5. department (choose strictly from: Cardiology, Neurology, Pulmonology, Orthopedics, Gastroenterology, General)

OUTPUT EXACTLY IN THIS JSON FORMAT:
{
  "chief_complaint": "string",
  "extracted_symptoms": ["string"],
  "duration_minutes": 120,
  "underlying_conditions": ["string"],
  "department": "string"
}"""

def extract_clinical_data(text: str) -> AIExtraction:
    """Sends the narrative to Llama 3 via Hugging Face and forces JSON output."""
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        max_tokens=300,
        temperature=0.1 # Very low temperature to prevent hallucinated formats
    )
    
    raw_response = response.choices[0].message.content.strip()
    
    # Safety net: Strip markdown formatting if Llama disobeys the prompt
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:-3]
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:-3]
        
    parsed_json = json.loads(raw_response)
    return AIExtraction(**parsed_json)
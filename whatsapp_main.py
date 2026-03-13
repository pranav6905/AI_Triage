from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from ai_service import generate_next_questions, Message # Assumes you have this from before

app = FastAPI()

# --- DATABASES ---
HOSPITALS = {"1": "City General Hospital", "2": "Rural Health Clinic"}
AVAILABLE_SLOTS = {"City General Hospital": ["10:00 AM", "02:00 PM"], "Rural Health Clinic": ["09:00 AM", "11:30 AM"]}

# --- GLOBAL SESSION MEMORY ---
user_sessions = {}

@app.post("/api/v1/whatsapp-booking")
async def whatsapp_bot(From: str = Form(...), Body: str = Form(...)):
    sender_id = From
    user_msg = Body.strip()
    response = MessagingResponse()

    # 1. INITIALIZE NEW USER
    if sender_id not in user_sessions:
        user_sessions[sender_id] = {
            "step": "choose_hospital",
            "history": [], # For the AI triage
            "collected_data": {}
        }
        msg = "Namaste! Welcome to Jeeva. 🏥\nSelect a hospital:\n1. City General\n2. Rural Clinic"
        response.message(msg)
        return PlainTextResponse(str(response), media_type="text/xml")

    session = user_sessions[sender_id]
    step = session["step"]

    # 2. STATE MACHINE LOGIC
    if step == "choose_hospital":
        if user_msg in HOSPITALS:
            session["collected_data"]["hospital"] = HOSPITALS[user_msg]
            session["step"] = "choose_slot"
            slots = AVAILABLE_SLOTS[HOSPITALS[user_msg]]
            response.message(f"Selected {HOSPITALS[user_msg]}.\nPick a slot:\n1. {slots[0]}\n2. {slots[1]}")
        else:
            response.message("Please reply with 1 or 2.")

    elif step == "choose_slot":
        h_name = session["collected_data"]["hospital"]
        session["collected_data"]["slot"] = AVAILABLE_SLOTS[h_name][int(user_msg)-1] if user_msg in ["1", "2"] else "TBD"
        session["step"] = "ask_problem"
        response.message("What is the main medical problem you are facing today?")

    elif step == "ask_problem":
        session["collected_data"]["problem"] = user_msg
        session["history"].append(Message(role="user", content=f"My problem is: {user_msg}"))
        session["step"] = "ask_duration"
        response.message("How long has this been happening? (e.g., 2 hours, 3 days)")

    elif step == "ask_duration":
        session["collected_data"]["duration"] = user_msg
        session["history"].append(Message(role="user", content=f"It has been happening for: {user_msg}"))
        session["step"] = "ask_ai_1"
        
        # CALL ENDPOINT 1 (Gemini)
        ai_questions = generate_next_questions(session["history"], "English")
        session["ai_pending_questions"] = ai_questions # Store the list of questions
        response.message(f"I understand. Let me ask a follow-up: \n\n{ai_questions[0]}")

    elif step == "ask_ai_1":
        session["history"].append(Message(role="user", content=user_msg))
        session["step"] = "ask_ai_2"
        # Ask the second AI question
        q2 = session["ai_pending_questions"][1] if len(session["ai_pending_questions"]) > 1 else "Are you feeling any dizziness?"
        response.message(q2)

    elif step == "ask_ai_2":
        session["history"].append(Message(role="user", content=user_msg))
        session["step"] = "ask_comorbidities"
        response.message("One last thing: Do you have any long-term health issues like Diabetes or Blood Pressure?")

    elif step == "ask_comorbidities":
        session["collected_data"]["comorbidities"] = user_msg
        
        # FINAL CONFIRMATION
        data = session["collected_data"]
        summary = (
            f"✅ *Slot Booked!*\n\n"
            f"📍 {data['hospital']}\n"
            f"⏰ {data['slot']}\n\n"
            f"The doctor has been notified about your: {data['problem']}.\n"
            f"Please show this message at the reception."
        )
        response.message(summary)
        del user_sessions[sender_id] # Clear session after booking

    return PlainTextResponse(str(response), media_type="text/xml")